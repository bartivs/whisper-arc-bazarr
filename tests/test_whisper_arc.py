"""Bazarr's raw PCM wire format, OpenVINO segment handling and request logs."""

import importlib.util
import io
import sys
import threading
import time
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

APP_PATH = Path(__file__).resolve().parents[1] / "src/app.py"
sys.path.insert(0, str(APP_PATH.parent))
spec = importlib.util.spec_from_file_location("whisper_arc", APP_PATH)
arc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arc)
client = TestClient(arc.app)


class FakePipeline:
    def __init__(self):
        self.calls = []

    def generate(self, audio, **options):
        self.calls.append((len(audio), options))
        return SimpleNamespace(language="en", chunks=[SimpleNamespace(
            start_ts=0.5, end_ts=min(2.0, len(audio) / arc.SAMPLE_RATE), text=" Hello world ",
        )])


def test_bazarr_connection_check_returns_version():
    result = client.get("/status")
    assert result.status_code == 200
    assert result.json()["version"]


def test_bazarr_asr_raw_pcm_and_segment_offsets(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    audio = np.zeros(arc.WINDOW_SAMPLES + arc.SAMPLE_RATE * 3, dtype="<i2").tobytes()
    result = client.post("/asr?encode=false&task=transcribe&language=en&output=srt", files={"audio_file": ("audio", audio)})
    assert result.status_code == 200
    assert "00:00:00,500 --> 00:00:02,000" in result.text
    assert "00:00:30,500 --> 00:00:32,000" in result.text
    assert len(model.calls) == 2
    assert model.calls[0][1] == {"task": "transcribe", "return_timestamps": True, "language": "<|en|>"}


def test_bazarr_language_detection(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    result = client.post("/detect-language?encode=false", files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert result.json() == {"language_code": "en", "detected_language": "en"}
    assert len(model.calls) == 1
    assert model.calls[0][0] == 16000


def test_wav_input_and_invalid_options(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    wav = io.BytesIO()
    with wave.open(wav, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 16000)
    result = client.post("/asr?encode=true&task=translate", files={"audio_file": ("audio.wav", wav.getvalue())})
    # ffmpeg is a runtime dependency; only exercise decoding where it is installed.
    assert result.status_code == 200
    assert model.calls[0][1]["task"] == "translate"
    assert client.post("/asr?task=unsupported", files={"audio_file": ("audio", b"\0\0")}).status_code == 400
    assert client.post("/detect-language", files={"audio_file": ("audio", b"\0")}).status_code == 400


@contextmanager
def busy_gate(seconds=0.2):
    """Hold the single pipeline like an in-flight request, then release it."""
    arc._gate.acquire()
    done = threading.Event()

    def release():
        time.sleep(seconds)
        arc._gate.release()
        done.set()

    thread = threading.Thread(target=release, daemon=True)
    thread.start()
    try:
        yield done
    finally:
        thread.join(5)
        if arc._gate.locked():
            arc._gate.release()
        assert not arc._gate.locked()


def test_serialization_and_limits(monkeypatch):
    monkeypatch.setattr(arc, "MAX_UPLOAD_BYTES", 4)
    assert client.post("/asr", files={"audio_file": ("audio", b"123456")}).status_code == 413


def test_extra_requests_queue_for_the_busy_pipeline_instead_of_failing(monkeypatch, caplog):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    audio = b"\0\0" * 16000
    with busy_gate():
        result = client.post(
            "/asr?encode=false&task=transcribe&language=en"
            "&video_file=/data/media/series/99/Season%2001/Show.S01E03.mkv",
            files={"audio_file": ("audio", audio)},
        )
    assert result.status_code == 200
    assert model.calls
    logs = caplog.text
    assert 'file="Show.S01E03.mkv"' in logs
    assert 'path="/data/media/series/99/Season 01/Show.S01E03.mkv"' in logs
    assert "asr queued" in logs and "waited_seconds=" in logs
    assert "asr start" in logs and "asr ok" in logs and "cues=1" in logs


def test_queue_timeout_and_full_queue_fail_with_retry_after(monkeypatch, caplog):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(arc, "GATE_WAIT_SECONDS", 0.0)
    audio = b"\0\0" * 16000
    with busy_gate():
        timed_out = client.post("/asr", params={"video_file": "/data/series/99/ep.mkv"},
                                files={"audio_file": ("audio", audio)})
    assert timed_out.status_code == 503
    assert timed_out.headers["retry-after"] == "60"
    assert not model.calls
    assert 'asr failed' in caplog.text and 'file="ep.mkv"' in caplog.text

    original = arc._waiters
    arc._waiters = threading.BoundedSemaphore(0)
    try:
        with busy_gate():
            full = client.post("/asr", params={"video_file": "/data/series/99/ep.mkv"},
                               files={"audio_file": ("audio", audio)})
    finally:
        arc._waiters = original
    assert full.status_code == 503
    assert full.headers["retry-after"] == "30"


def test_logs_name_the_file_on_validation_failure_and_empty_transcript(monkeypatch, caplog):
    rejected = client.post("/asr?task=unsupported&video_file=/data/series/99/Show.S01E04.mkv",
                           files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert rejected.status_code == 400

    class Silent(FakePipeline):
        def generate(self, audio, **options):
            return SimpleNamespace(language="en", chunks=[])

    monkeypatch.setattr(arc, "pipeline", lambda: Silent())
    empty = client.post("/asr?encode=false&video_file=/data/series/99/Show.S01E05.mkv",
                        files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert empty.status_code == 200
    logs = caplog.text
    assert 'asr failed file="Show.S01E04.mkv"' in logs and "status=400" in logs
    assert 'asr empty file="Show.S01E05.mkv"' in logs and "cues=0" in logs


def test_detect_logs_the_file_name(monkeypatch, caplog):
    monkeypatch.setattr(arc, "pipeline", lambda: FakePipeline())
    result = client.post("/detect-language?encode=false&video_file=/data/series/99/nested/Show.S01E06.mkv",
                         files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert result.status_code == 200
    logs = caplog.text
    assert 'detect ok file="Show.S01E06.mkv"' in logs and "language=en" in logs
