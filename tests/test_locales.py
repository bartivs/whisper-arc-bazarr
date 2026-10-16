"""POSIX locale configuration and resilient structured translation contract."""

import json
import threading
import time

import pytest

from test_whisper_arc import arc, client, FakePipeline
import translation
from languages import parse_locale


@pytest.mark.parametrize("value,unix,label", [
    ("es_MX.UTF-8", "es_MX.UTF-8", "Spanish (Mexico)"),
    ("es_419.UTF-8", "es_419.UTF-8", "Spanish (Latin America)"),
    ("es-419", "es_419.UTF-8", "Spanish (Latin America)"),
    ("es-MX", "es_MX.UTF-8", "Spanish (Mexico)"),
    ("es_es.utf8", "es_ES.UTF-8", "Spanish (Spain)"),
    ("en_US", "en_US.UTF-8", "English (United States)"),
    ("en", "en", "English"),
])
def test_locale_aliases(value, unix, label):
    locale = parse_locale(value)
    assert locale.unix == unix
    assert locale.label == label


@pytest.mark.parametrize("value", ["C", "es_XX.UTF-8", "en_MX", "es_MX@evil", "es_MX.ISO-8859-1", "es\nMX"])
def test_bad_locale_rejected(value):
    with pytest.raises(ValueError):
        parse_locale(value)


def test_posix_api_language_and_target_are_normalized(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(translation, "translate_texts", lambda texts, source_code="en": ["Hola" for _ in texts])
    response = client.post("/asr", params={"language": "en_US.UTF-8", "task": "translate", "target_language": "es_MX.UTF-8"},
                           files={"audio_file": ("test.pcm", b"\0\0" * 16000)})
    assert response.status_code == 200
    assert "Hola" in response.text
    assert model.calls[-1][1]["language"] == "<|en|>"
    assert client.get("/status").json()["target_locale"] == "es_MX.UTF-8"
    response = client.post("/asr?language=en&task=translate&target_language=es_ES.UTF-8",
                           files={"audio_file": ("test.pcm", b"\0\0" * 16000)})
    assert response.status_code == 400


@pytest.mark.parametrize("cue", [
    {"i": 0, "extra": "missing text"}, {"text": "Hola", "extra": "missing index"},
    {"i": True, "text": "Hola"}, {"i": 0.8, "text": "Hola"}, {"i": "0", "text": "Hola"},
    {"i": 0, "text": None}, [0, "Hola"],
])
def test_malformed_cue_shapes_never_escape_validation(cue):
    body = {"choices": [{"message": {"content": json.dumps({"cues": [cue]})}}]}
    with pytest.raises(ValueError):
        translation._extract_cues(body, ["Hello"])


def test_structured_regional_prompt_and_private_named_logs(monkeypatch, caplog):
    payloads = []
    def post(payload):
        payloads.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"cues":[{"i":0,"text":"Hola"}]}'}}]}
    monkeypatch.setattr(translation, "_post", post)
    monkeypatch.setattr(translation, "ENDPOINT", "http://test/v1")
    monkeypatch.setattr(translation, "API_KEY", "fake-token")
    source = b"1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    result = translation.translate_srt(source, media_name="Show.S01E03.mkv")
    assert b"00:00:01,000\nHola" in result
    assert payloads[0]["response_format"]["type"] == "json_schema"
    assert payloads[0]["response_format"]["json_schema"]["strict"] is True
    assert "Spanish (Mexico)" in payloads[0]["messages"][0]["content"]
    assert "Show.S01E03.mkv" not in json.dumps(payloads)
    assert 'translation ok file="Show.S01E03.mkv" batch=1' in caplog.text
    assert "engine=litellm" in caplog.text
    assert "fake-token" not in caplog.text


def test_transient_failure_retries_without_exposing_subtitles(monkeypatch, caplog):
    calls = []
    def post(payload):
        calls.append(1)
        if len(calls) == 1:
            raise translation.ProxyError("http_429")
        return {"choices": [{"message": {"content": '{"cues":[{"i":0,"text":"Hola"}]}'}}]}
    monkeypatch.setattr(translation, "_post", post)
    monkeypatch.setattr(translation, "RETRY_DELAY", 0)
    assert translation.proxy_translate(["PRIVATE SUBTITLE"]) == ["Hola"]
    assert len(calls) == 2
    assert "http_429" in caplog.text and "PRIVATE SUBTITLE" not in caplog.text


def test_fifo_queue():
    order = []
    arc._gate.acquire()
    threads = []
    try:
        for index in range(3):
            def work(index=index):
                with arc.exclusive():
                    order.append(index)
                    time.sleep(0.01)
            thread = threading.Thread(target=work)
            thread.start()
            threads.append(thread)
            deadline = time.monotonic() + 2
            while len(arc._queue) < index + 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(arc._queue) == index + 1
    finally:
        arc._gate.release()
        for thread in threads:
            thread.join(3)
    assert order == [0, 1, 2]
    assert not arc._queue and not arc._gate.locked()
