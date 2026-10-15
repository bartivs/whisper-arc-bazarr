"""Run without GPU/models; real model smoke tests are documented separately."""

import importlib.util
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import translation
from test_whisper_arc import arc, client, FakePipeline


def test_spanish_api_keeps_english_asr_and_cue_timings(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(translation, "translate_texts", lambda texts: ["Hola mundo" for _ in texts])
    response = client.post("/asr?task=translate&language=en&target_language=es", files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert response.status_code == 200
    assert "00:00:00,500 --> 00:00:01,000\nHola mundo" in response.text
    assert model.calls[0][1]["task"] == "transcribe"
    assert model.calls[0][1]["language"] == "<|en|>"


@pytest.mark.parametrize("query", ["language=fr&task=translate", "language=en&task=transcribe", "language=en&task=translate&target_language=de"])
def test_invalid_pairs_rejected(query):
    if "target_language" not in query:
        query += "&target_language=es"
    assert client.post("/asr?" + query, files={"audio_file": ("audio", b"\0\0")}).status_code == 400


def test_translation_failure_does_not_return_english(monkeypatch):
    monkeypatch.setattr(arc, "pipeline", lambda: FakePipeline())
    monkeypatch.setattr(translation, "translate_texts", lambda texts: [])
    response = client.post("/asr?task=translate&language=en&target_language=es", files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert response.status_code == 503
    assert "Hello world" not in response.text


def test_srt_batching_preserves_every_cue(monkeypatch):
    calls = []
    def translate(texts):
        calls.append(len(texts))
        return ["Texto traducido" for _ in texts]
    monkeypatch.setattr(translation, "translate_texts", translate)
    data = "\n".join(f"{i}\n00:00:00,100 --> 00:00:00,900\nEnglish {i}\n" for i in range(1, 34)).encode()
    result = translation.translate_srt(data).decode()
    assert calls == [16, 16, 1]
    assert result.count("00:00:00,100 --> 00:00:00,900") == 33
    assert "33\n00:00:00,100" in result
    assert "English" not in result
    assert translation.translate_srt(b"") == b""
    with pytest.raises(ValueError):
        translation.translate_srt(b"not srt")


def hook_module():
    spec = importlib.util.spec_from_file_location("patch_provider", ROOT / "bazarr/patch_provider.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hook_rejects_unknown_and_preserves_idempotence(monkeypatch):
    module = hook_module()
    with pytest.raises(RuntimeError):
        module.patch("unknown upstream code")
    source = '''class Provider:
    def query(self, language, sub):
        if sub.task == "translate":
            if language.alpha3 != "eng":
                return None
        return sub
    def download(self, subtitle, input_language, video_name, out):
        if subtitle.task == "error":
            return
        if subtitle.task == "transcribe":
            output_language = subtitle.audio_language
        else:
            output_language = "eng"
        r = self.session.post(f"{self.endpoint}/asr",
                              params={'task': subtitle.task, 'language': input_language, 'output': 'srt', 'encode': 'false',
                                      'video_file': video_name},
                              files={'audio_file': out},
                              timeout=(5, 3600))
        subtitle.content = r.content
'''
    monkeypatch.setattr(module, "UPSTREAM_SHA256", hashlib.sha256(source.encode()).hexdigest())
    patched = module.patch(source)
    assert module.patch(patched) == patched
    assert "target_language" in patched
    with pytest.raises(RuntimeError):
        module.patch(patched.replace("return sub", "return None"))
    # Execute just this small synthetic provider to check endpoint scoping.
    from types import SimpleNamespace
    env = {"os": __import__("os"), "ConfigurationError": ValueError}
    exec(patched, env)
    monkeypatch.setenv("WHISPER_TRANSLATION_ENDPOINT", "http://whisper-arc:9000")
    provider = env["Provider"]()
    provider.endpoint = "http://subgen:9000"
    sub = SimpleNamespace(task="translate", audio_language="eng")
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is None
    sub.language = SimpleNamespace(alpha3="spa")
    with pytest.raises(ValueError, match="configured Arc endpoint"):
        provider.download(sub, "en", None, b"pcm")
    provider.endpoint = "http://whisper-arc:9000"
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is sub
    sub.audio_language = "fra"
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is None
