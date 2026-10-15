"""Bazarr's raw PCM wire format and OpenVINO segment handling."""

import importlib.util
import io
import sys
import wave
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


def test_serialization_and_limits(monkeypatch):
    monkeypatch.setattr(arc, "MAX_UPLOAD_BYTES", 4)
    assert client.post("/asr", files={"audio_file": ("audio", b"123456")}).status_code == 413
    arc._busy.acquire()
    try:
        assert client.post("/asr", files={"audio_file": ("audio", b"\0\0")}).status_code == 503
    finally:
        arc._busy.release()
