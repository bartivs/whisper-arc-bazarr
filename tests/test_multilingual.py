"""Per-request multilingual output without cross-request state or unsafe fallback."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_whisper_arc import arc, client, FakePipeline
import languages
import translation


@pytest.mark.parametrize("value,label", [
    ("fr_FR.UTF-8", "French (France)"), ("pt-BR", "Portuguese (Brazil)"),
    ("de_DE", "German (Germany)"), ("ja", "Japanese"), ("zh-TW", "Chinese (Taiwan)"),
    ("ar", "Arabic"), ("hi_IN", "Hindi (IN)"), ("ko_KR", "Korean (South Korea)"),
])
def test_multilingual_locales(value, label):
    assert languages.parse_locale(value).label == label


@pytest.mark.parametrize("value", ["xx", "und", "en_999", "fr_XX", "fr;ignore instructions", "zh_Hant"])
def test_invalid_target_locales(value):
    with pytest.raises(ValueError):
        languages.parse_locale(value)


def test_configured_locale_is_only_a_default(monkeypatch):
    monkeypatch.setattr(languages, "TARGET_LOCALE", languages.parse_locale("pt_BR"))
    assert languages.target_locale("pt").unix == "pt_BR.UTF-8"
    assert languages.target_locale("pt-PT").unix == "pt_PT.UTF-8"
    assert languages.target_locale("fr").unix == "fr"


def test_non_english_configuration_starts_without_gpu_or_models():
    env = {**os.environ, "TRANSLATION_SOURCE_LOCALE": "ja_JP.UTF-8",
           "TRANSLATION_TARGET_LOCALE": "fr_FR.UTF-8"}
    result = subprocess.run(
        [sys.executable, "-c", "import languages; print(languages.SOURCE_LOCALE.label, languages.TARGET_LOCALE.label)"],
        cwd=Path(__file__).resolve().parents[1] / "src", env=env,
        capture_output=True, text=True, check=True,
    )
    assert "Japanese (Japan) French (France)" in result.stdout


@pytest.mark.parametrize("source,target,expected", [
    ("en", "fr_FR.UTF-8", "French (France)"),
    ("de", "ja", "Japanese"), ("ja", "pt-BR", "Portuguese (Brazil)"),
    ("ar", "zh-TW", "Chinese (Taiwan)"), ("fr", "en", "English"),
    ("zh", "ko", "Korean"),
])
def test_any_source_to_requested_target_api(monkeypatch, caplog, source, target, expected):
    model = FakePipeline()
    def generate(audio, **options):
        model.calls.append((len(audio), options))
        return SimpleNamespace(language=source, chunks=[SimpleNamespace(start_ts=0.5, end_ts=1, text="Dialogue")])
    model.generate = generate
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(translation, "ENDPOINT", "http://test/v1")
    monkeypatch.setattr(translation, "API_KEY", "test-only")
    payloads = []
    def post(payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": json.dumps({"cues": [{"i": 0, "text": "翻訳された台詞"}]})}}]}
    monkeypatch.setattr(translation, "_post", post)
    result = client.post("/asr", params={"task": "translate", "target_language": target},
                         files={"audio_file": ("clip.pcm", b"\0\0" * 16000)})
    assert result.status_code == 200
    assert "00:00:00,500 --> 00:00:01,000\n翻訳された台詞" in result.text
    prompt = payloads[0]["messages"][0]["content"]
    assert f"each {languages.language_name(source)} subtitle cue" in prompt
    assert f"into natural {expected}" in prompt
    assert "Latin American" not in prompt
    assert model.calls[-1][1]["task"] == "transcribe"
    assert model.calls[-1][1]["language"] == f"<|{source}|>"
    assert f"target={languages.target_locale(target).unix}" in caplog.text
    assert "test-only" not in caplog.text


@pytest.mark.parametrize("request_language,expected", [("", "de"), ("ja_JP", "ja")])
def test_unknown_acoustic_source_uses_request_or_configured_fallback(monkeypatch, request_language, expected):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(arc, "detect", lambda pcm: {"language_code": "und"})
    monkeypatch.setattr(arc, "SOURCE_LOCALE", languages.parse_locale("de_DE"))
    calls = []
    def translate(data, media_name, source_code, target_locale):
        calls.append((source_code, target_locale.language))
        return data
    monkeypatch.setattr(translation, "translate_srt", translate)
    result = client.post("/asr", params={"task": "translate", "target_language": "fr", "language": request_language},
                         files={"audio_file": ("clip.pcm", b"\0\0" * 16000)})
    assert result.status_code == 200
    assert calls == [(expected, "fr")]
    assert model.calls[-1][1]["language"] == f"<|{expected}|>"


@pytest.mark.parametrize("params", [
    {"task": "translate", "target_language": "xx"},
    {"task": "translate", "target_language": "fr_XX"},
    {"task": "transcribe", "target_language": "ja"},
    {"task": "translate", "target_language": "ja", "language": "xx"},
])
def test_invalid_translation_requests_fail_before_inference(monkeypatch, params):
    monkeypatch.setattr(arc, "pipeline", lambda: pytest.fail("invalid request reached inference"))
    response = client.post("/asr", params=params, files={"audio_file": ("clip", b"\0\0" * 16000)})
    assert response.status_code == 400


@pytest.mark.parametrize("language", ["jw", "jv", "jv_ID"])
def test_javanese_source_uses_whisper_legacy_token(monkeypatch, language):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    result = client.post("/asr", params={"language": language}, files={"audio_file": ("clip", b"\0\0" * 16000)})
    assert result.status_code == 200
    assert model.calls[-1][1]["language"] == "<|jw|>"


def test_native_english_translation_does_not_call_text_engine(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(translation, "translate_srt", lambda *args, **kwargs: pytest.fail("native translation uses Whisper"))
    response = client.post("/asr?task=translate&language=fr", files={"audio_file": ("clip", b"\0\0" * 16000)})
    assert response.status_code == 200
    assert model.calls[-1][1]["task"] == "translate"
    assert model.calls[-1][1]["language"] == "<|fr|>"


@pytest.mark.parametrize("source,target", [("en", "fr"), ("ja", "es"), ("de", "pt_BR"), ("fr", "en")])
@pytest.mark.parametrize("configured", [True, False])
def test_unsupported_offline_pairs_fail_closed(monkeypatch, source, target, configured):
    monkeypatch.setattr(translation, "proxy_configured", lambda: configured)
    monkeypatch.setattr(translation, "LOCAL_FALLBACK", True)
    monkeypatch.setattr(translation, "local_available", lambda: True)
    monkeypatch.setattr(translation, "proxy_translate", lambda *args: (_ for _ in ()).throw(RuntimeError("unavailable")))
    monkeypatch.setattr(translation, "local_translate", lambda *args: pytest.fail("wrong language model"))
    with pytest.raises(RuntimeError):
        translation.translate_texts(["Dialogue"], source, languages.parse_locale(target))


def test_offline_pair_is_fixed_even_when_config_changes(monkeypatch):
    monkeypatch.setattr(translation, "SOURCE_LOCALE", languages.parse_locale("de"))
    monkeypatch.setattr(translation, "TARGET_LOCALE", languages.parse_locale("fr"))
    monkeypatch.setattr(translation, "proxy_configured", lambda: False)
    monkeypatch.setattr(translation, "local_translate", lambda texts: ["Hola"])
    assert translation.translate_texts(["Hello"], "en", languages.parse_locale("es")) == ["Hola"]
    with pytest.raises(RuntimeError):
        translation.translate_texts(["Hello"], "en")


def test_same_language_does_not_require_a_translation_engine(monkeypatch):
    monkeypatch.setattr(translation, "proxy_configured", lambda: pytest.fail("no engine needed"))
    assert translation.translate_texts(["Bonjour"], "fr", languages.parse_locale("fr_FR")) == ["Bonjour"]


def test_failed_non_spanish_api_returns_no_subtitles(monkeypatch):
    monkeypatch.setattr(arc, "pipeline", lambda: FakePipeline())
    monkeypatch.setattr(translation, "proxy_configured", lambda: False)
    monkeypatch.setattr(translation, "local_translate", lambda texts: pytest.fail("wrong model"))
    result = client.post("/asr?task=translate&target_language=fr", files={"audio_file": ("clip", b"\0\0" * 16000)})
    assert result.status_code == 503
    assert result.json()["detail"] == "Text translation failed; no subtitles returned"
    assert "Hello world" not in result.text


@pytest.mark.parametrize("target,latin_style", [("es_419", True), ("es_MX", True), ("es_ES", False), ("fr", False)])
def test_spanish_prompt_is_scoped_to_the_requested_region(monkeypatch, target, latin_style):
    payloads = []
    def post(payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": '{"cues":[{"i":0,"text":"Translated"}]}'}}]}
    monkeypatch.setattr(translation, "_post", post)
    translation.proxy_translate(["Hello"], "en", languages.parse_locale(target))
    assert ("Latin American" in payloads[0]["messages"][0]["content"]) is latin_style


def test_concurrent_targets_do_not_leak_between_requests(monkeypatch):
    monkeypatch.setattr(translation, "proxy_configured", lambda: True)
    def post(payload):
        prompt = payload["messages"][0]["content"]
        text = "Bonjour" if "natural French" in prompt else "こんにちは"
        return {"choices": [{"message": {"content": json.dumps({"cues": [{"i": 0, "text": text}]})}}]}
    monkeypatch.setattr(translation, "_post", post)
    original = translation._CONTEXT.get()
    data = b"1\n00:00:00,000 --> 00:00:01,000\nHello\n"
    targets = ["fr", "ja"] * 8
    def run(target):
        return translation.translate_srt(data, source_code="en", target_locale=languages.parse_locale(target)).decode()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, targets))
    for target, result in zip(targets, results):
        assert ("Bonjour" if target == "fr" else "こんにちは") in result
        assert "00:00:00,000 --> 00:00:01,000" in result
    assert translation._CONTEXT.get() == original


@pytest.mark.parametrize("hint,acoustic,text,expected", [
    ("", "ja", "", "ja"), ("und", "de", "", "de"), ("", "jw", "", "jv"),
    ("und", "und", "", "en"), ("", "xx", "", "en"),
    ("", "und", "the and you have to do what is in the house and what you do", "en"),
])
def test_source_resolution_rejects_ambiguous_codes(hint, acoustic, text, expected):
    assert languages.resolve_source(hint, acoustic, text)[0] == expected
