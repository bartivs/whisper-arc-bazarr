"""Small Bazarr Whisper-provider adapter backed by OpenVINO GenAI on Intel Arc.

Bazarr uploads 16 kHz mono s16le PCM with encode=false; no media paths, API
credentials or host ports are required. One request runs at a time to protect
arr's 4 GiB LXC. Never fall back to CPU silently when the GPU is unavailable.
"""

import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR", "/models")
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 30 * SAMPLE_RATE
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_PCM_BYTES = 512 * 1024 * 1024

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_pipeline = None
_busy = threading.Lock()


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
    return {"version": "whisper-arc-openvino-base-1"}


@app.get("/health")
def health():
    import openvino as ov

    if "GPU" not in ov.Core().available_devices or not Path(MODEL_DIR, "generation_config.json").is_file():
        raise HTTPException(503, "Intel GPU or Whisper model unavailable")
    return {"device": ov.Core().get_property("GPU", "FULL_DEVICE_NAME"), "model": "base"}


@app.post("/asr")
def asr(audio_file: UploadFile = File(...), task: str = "transcribe", language: str = "", output: str = "srt", encode: str = "false", target_language: str = ""):
    if task not in ("transcribe", "translate") or output != "srt":
        raise HTTPException(400, "Only transcribe/translate with SRT output is supported")
    if language and (len(language) != 2 or not language.isalpha()):
        raise HTTPException(400, "language must be an ISO 639-1 code")
    if target_language and not (target_language == "es" and language == "en" and task == "translate"):
        raise HTTPException(400, "Text translation supports English to Spanish only")
    if not _busy.acquire(blocking=False):
        raise HTTPException(503, "Whisper is processing another request")
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-arc-") as directory:
            # Whisper cannot target Spanish. First transcribe English, then
            # translate cue text only; keep all generated SRT timings unchanged.
            data = transcribe(pcm_path(audio_file, encode, directory), "transcribe" if target_language else task, language or "")
            if target_language:
                from translation import translate_srt

                try:
                    data = translate_srt(data)
                except (RuntimeError, ValueError, OSError) as exc:
                    raise HTTPException(503, "Local Spanish translation failed; no subtitles returned") from exc
        return Response(data, media_type="application/x-subrip")
    finally:
        _busy.release()


@app.post("/detect-language")
def detect_language(audio_file: UploadFile = File(...), encode: str = "false"):
    if not _busy.acquire(blocking=False):
        raise HTTPException(503, "Whisper is processing another request")
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-arc-") as directory:
            result = detect(pcm_path(audio_file, encode, directory))
        return JSONResponse(result)
    finally:
        _busy.release()
