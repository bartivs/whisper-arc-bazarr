"""Small Bazarr Whisper-provider adapter backed by OpenVINO GenAI on Intel Arc.

Bazarr uploads 16 kHz mono s16le PCM with encode=false; no media paths, API
credentials or host ports are required. One request runs at a time to protect
arr's 4 GiB LXC. Never fall back to CPU silently when the GPU is unavailable.

Every request logs the media file name Bazarr passes as the `video_file` query
parameter, so a failing episode can be found by grepping the container logs for
its file name. Extra requests wait for the single pipeline instead of failing,
because Bazarr turns a 503 into a failed download (and the provider hook raises
on non-2xx responses).
"""

import json
import logging
import re
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from languages import SOURCE_LOCALE, TARGET_LOCALE, parse_locale, resolve_source
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_PCM_BYTES = 512 * 1024 * 1024
# How long a queued request waits for the busy pipeline before answering 503,
# and how many requests may queue behind it at once.
GATE_WAIT_SECONDS = max(0.0, float(os.environ.get("WHISPER_GATE_WAIT_SECONDS", "900")))
MAX_WAITERS = max(0, int(os.environ.get("WHISPER_MAX_WAITERS", "4")))

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_pipeline = None
_gate = threading.Lock()
_waiters = threading.BoundedSemaphore(MAX_WAITERS)
_queue = deque()
_queue_changed = threading.Condition()

LOG = logging.getLogger("whisper_arc")
LOG.setLevel(os.environ.get("WHISPER_LOG_LEVEL", "INFO").upper())
if not LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s whisper_arc %(message)s"))
    LOG.addHandler(_handler)
LOG.propagate = False


_FILENAME_LANGUAGES = {
    "german": "de", "deutsch": "de", "ger": "de", "deu": "de",
    "english": "en", "eng": "en",
    "spanish": "es", "espanol": "es", "spa": "es",
    "french": "fr", "fre": "fr", "fra": "fr",
    "italian": "it", "ita": "it", "portuguese": "pt", "por": "pt",
    "dutch": "nl", "nld": "nl", "russian": "ru", "rus": "ru",
}


def language_hint(text):
    """Container/file-name language hint only; the supplied path is never opened."""
    for token in re.split(r"[^a-z]+", (text or "").lower()):
        if token in _FILENAME_LANGUAGES:
            return _FILENAME_LANGUAGES[token]
    return ""


def source_label(video_file, upload):
    """Return log-safe `(name, path)` for the media file; never used for IO."""
    raw = video_file or getattr(upload, "filename", "") or ""
    raw = "".join(character for character in raw if character.isprintable())[:500]
    return (Path(raw).name if raw else "unknown")[:200], raw


def log_event(event, *, name, path="", **fields):
    """One greppable line per request transition, keyed by media file name."""
    parts = [f'file={json.dumps(name, ensure_ascii=False)}']
    if path and path != name:
        parts.append(f'path={json.dumps(path, ensure_ascii=False)}')
    parts.extend(f"{key}={value}" for key, value in fields.items() if value is not None)
    LOG.info("%s %s", event, " ".join(parts))


@contextmanager
def exclusive():
    """Run one request at a time, queueing bounded waiters instead of dropping them."""
    waited = 0.0
    with _queue_changed:
        if _queue or not _gate.acquire(blocking=False):
            if not _waiters.acquire(blocking=False):
                raise HTTPException(503, "Whisper queue is full", headers={"Retry-After": "30"})
            ticket = object()
            _queue.append(ticket)
            started = time.monotonic()
            try:
                while _queue[0] is not ticket or not _gate.acquire(blocking=False):
                    remaining = GATE_WAIT_SECONDS - (time.monotonic() - started)
                    if remaining <= 0:
                        raise HTTPException(503, "Whisper queue wait timed out", headers={"Retry-After": "60"})
                    _queue_changed.wait(min(remaining, 0.1))
                waited = time.monotonic() - started
            finally:
                _queue.remove(ticket)
                _waiters.release()
                _queue_changed.notify_all()
    try:
        yield waited
    finally:
        with _queue_changed:
            _gate.release()
            _queue_changed.notify_all()


def failure_detail(exc):
    """Bounded HTTP status/detail for logs; never raises or exposes payloads."""
    if isinstance(exc, HTTPException):
        return exc.status_code, str(exc.detail)[:200]
    return 500, type(exc).__name__


def pipeline():
    global _pipeline
    if _pipeline is None:
        import openvino as ov
        import openvino_genai as genai

        if "GPU" not in ov.Core().available_devices:
            raise RuntimeError("Intel GPU is not available; refusing CPU fallback")
        _pipeline = genai.WhisperPipeline(MODEL_DIR, "GPU")
    return _pipeline


def save_upload(upload, path):
    total = 0
    with open(path, "wb") as target:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Audio upload exceeds 512 MiB")
            target.write(chunk)
    if not total:
        raise HTTPException(400, "Empty audio upload")


def pcm_path(upload, encode, directory):
    source = Path(directory) / "upload"
    save_upload(upload, source)
    if encode.lower() == "false":
        pcm = source
    elif encode.lower() == "true":
        pcm = Path(directory) / "audio.pcm"
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", str(pcm)],
                check=True, capture_output=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(400, "Could not decode audio") from exc
    else:
        raise HTTPException(400, "encode must be true or false")
    size = pcm.stat().st_size
    if not 0 < size <= MAX_PCM_BYTES or size % 2:
        raise HTTPException(413 if size > MAX_PCM_BYTES else 400, "Invalid PCM audio length")
    return pcm


def seconds_to_srt(seconds):
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def transcribe(pcm, task, language):
    samples = np.memmap(pcm, dtype="<i2", mode="r")
    lines = []
    counter = 0
    for start in range(0, len(samples), WINDOW_SAMPLES):
        audio = np.asarray(samples[start:start + WINDOW_SAMPLES], dtype=np.float32) / 32768.0
        options = {"task": task, "return_timestamps": True}
        if language:
            options["language"] = f"<|{language}|>"
        result = pipeline().generate(audio, **options)
        offset = start / SAMPLE_RATE
        for segment in result.chunks:
            text = segment.text.strip()
            if not text:
                continue
            begin = offset + max(0.0, float(segment.start_ts))
            end = offset + min(len(audio) / SAMPLE_RATE, float(segment.end_ts))
            if end <= begin:
                continue
            counter += 1
            lines.append(f"{counter}\n{seconds_to_srt(begin)} --> {seconds_to_srt(end)}\n{text}\n")
    return "\n".join(lines).encode("utf-8")


def cue_count(data):
    return data.count(b"\n\n") + 1 if data.strip() else 0


def detect(pcm):
    samples = np.memmap(pcm, dtype="<i2", mode="r")
    audio = np.asarray(samples[:WINDOW_SAMPLES], dtype=np.float32) / 32768.0
    result = pipeline().generate(audio, task="transcribe")
    # Bazarr requires language_code (not detected_language as a code). GenAI
    # reports the automatically detected ISO 639-1 code in result.language.
    code = (result.language or "und").strip("<>| ").lower()
    if not code.isalpha() or len(code) != 2:
        code = "und"
    return {"language_code": code, "detected_language": code}


@app.get("/status")
def status():
    """Bazarr's Test Connection button expects JSON containing a version string."""
    return {"version": "whisper-arc-openvino-base-3", "source_locale": SOURCE_LOCALE.unix,
            "target_locale": TARGET_LOCALE.unix}


@app.get("/health")
def health():
    import openvino as ov

    if "GPU" not in ov.Core().available_devices or not Path(MODEL_DIR, "generation_config.json").is_file():
        raise HTTPException(503, "Intel GPU or Whisper model unavailable")
    return {"device": ov.Core().get_property("GPU", "FULL_DEVICE_NAME"), "model": "base",
            "source_locale": SOURCE_LOCALE.unix, "target_locale": TARGET_LOCALE.unix}


@app.post("/asr")
def asr(audio_file: UploadFile = File(...), task: str = "transcribe", language: str = "", output: str = "srt", encode: str = "false", target_language: str = "", video_file: str = ""):
    name, path = source_label(video_file, audio_file)
    started = time.monotonic()
    log_event("asr start", name=name, path=path, task=task, language=language or "-",
              target=target_language or "-", encode=encode)
    try:
        try:
            if language and len(language) > 2:
                language = parse_locale(language).language
            if target_language and len(target_language) > 2:
                locale = parse_locale(target_language)
                if locale.region and locale != TARGET_LOCALE:
                    raise ValueError("Requested target locale differs from configured target locale")
                target_language = locale.language
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if task not in ("transcribe", "translate") or output != "srt":
            raise HTTPException(400, "Only transcribe/translate with SRT output is supported")
        if language and (len(language) != 2 or not language.isalpha()):
            raise HTTPException(400, "language must be an ISO 639-1 code")
        if target_language and (target_language != "es" or task != "translate"):
            raise HTTPException(400, "Text translation targets Spanish with task=translate only")
        with exclusive() as waited:
            if waited:
                log_event("asr queued", name=name, path=path, waited_seconds=round(waited, 1))
            with tempfile.TemporaryDirectory(prefix="whisper-arc-") as directory:
                # Whisper cannot target Spanish. First transcribe English, then
                # translate cue text only; keep all generated SRT timings unchanged.
                pcm = pcm_path(audio_file, encode, directory)
                source = language or ""
                acoustic = ""
                if target_language:
                    # Any source language is allowed: the file may be a German
                    # dub, so resolve the source instead of assuming English.
                    acoustic = detect(pcm)["language_code"]
                    source, evidence = resolve_source(
                        container_tag=language_hint(name), whisper_code=acoustic, fallback="en")
                    log_event("language resolved", name=name, source_lang=source,
                              acoustic=acoustic or "-", evidence=evidence)
                data = transcribe(pcm, "transcribe" if target_language else task, source)
                log_event("asr transcribed", name=name, cues=cue_count(data), source_lang=source or "auto")
                if target_language:
                    from translation import translate_srt

                    try:
                        # Cross-check the transcript itself; a lexical win means the
                        # acoustic guess and the stored tag were both wrong.
                        source, evidence = resolve_source(
                            container_tag=language_hint(name), whisper_code=acoustic,
                            text=data.decode("utf-8", "ignore"),
                            fallback=source or "en")
                        data = translate_srt(data, media_name=name, source_code=source)
                        log_event("translation ok", name=name, source_lang=source, evidence=evidence)
                    except (RuntimeError, ValueError, OSError) as exc:
                        log_event("translation failed", name=name, reason=type(exc).__name__)
                        raise HTTPException(503, "Spanish translation failed; no subtitles returned") from exc
        cues = cue_count(data)
        log_event("asr ok" if cues else "asr empty", name=name, path=path, cues=cues,
                  bytes_out=len(data), seconds=round(time.monotonic() - started, 1))
        return Response(data, media_type="application/x-subrip")
    except HTTPException as exc:
        log_event("asr failed", name=name, path=path, status=exc.status_code, reason=str(exc.detail)[:200],
                  seconds=round(time.monotonic() - started, 1))
        raise
    except Exception as exc:
        LOG.exception("asr error file=\"%s\" path=\"%s\"", name, path)
        status, reason = failure_detail(exc)
        log_event("asr failed", name=name, path=path, status=status, reason=reason,
                  seconds=round(time.monotonic() - started, 1))
        raise


@app.post("/detect-language")
def detect_language(audio_file: UploadFile = File(...), encode: str = "false", video_file: str = ""):
    name, path = source_label(video_file, audio_file)
    started = time.monotonic()
    log_event("detect start", name=name, path=path, encode=encode)
    try:
        with exclusive() as waited:
            if waited:
                log_event("detect queued", name=name, path=path, waited_seconds=round(waited, 1))
            with tempfile.TemporaryDirectory(prefix="whisper-arc-") as directory:
                result = detect(pcm_path(audio_file, encode, directory))
        log_event("detect ok", name=name, path=path, language=result["language_code"],
                  seconds=round(time.monotonic() - started, 1))
        return JSONResponse(result)
    except HTTPException as exc:
        log_event("detect failed", name=name, path=path, status=exc.status_code, reason=str(exc.detail)[:200],
                  seconds=round(time.monotonic() - started, 1))
        raise
    except Exception as exc:
        LOG.exception("detect error file=\"%s\" path=\"%s\"", name, path)
        status, reason = failure_detail(exc)
        log_event("detect failed", name=name, path=path, status=status, reason=reason,
                  seconds=round(time.monotonic() - started, 1))
        raise
