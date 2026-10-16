"""Translation engine tests: LiteLLM proxy path, validation, and local fallback.

Run without GPU, network or model weights; the proxy and local engine are faked.
"""

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import translation
from test_whisper_arc import arc, client, FakePipeline


@pytest.fixture(autouse=True)
def proxy_env(monkeypatch):
    monkeypatch.setattr(translation, "ENDPOINT", "http://litellm.example/v1")
    monkeypatch.setattr(translation, "API_KEY", "sk-test-only")  # pragma: allowlist secret
    monkeypatch.setattr(translation, "MODEL", "subtitle-translate")
    monkeypatch.setattr(translation, "LOCAL_FALLBACK", True)
    monkeypatch.setattr(translation, "local_available", lambda: True)
    monkeypatch.setattr(translation, "RETRY_DELAY", 0)


def reply(cues):
    return {"choices": [{"message": {"content": json.dumps({"cues": cues})}}]}


def test_proxy_batches_are_index_validated(monkeypatch):
    calls = []

    def fake_post(payload):
        cues = json.loads(payload["messages"][1]["content"])["cues"]
        calls.append(payload["model"])
        return reply([{"i": c["i"], "text": f"ES:{c['text']}"} for c in cues])

    monkeypatch.setattr(translation, "_post", fake_post)
    assert translation.translate_texts(["Hello", "World"]) == ["ES:Hello", "ES:World"]
    assert calls == ["subtitle-translate"]


def test_reordered_cues_are_restored_by_index(monkeypatch):
    """Cue indices, not reply order, decide where text lands."""
    monkeypatch.setattr(translation, "_post", lambda payload: reply([{"i": 1, "text": "mundo"}, {"i": 0, "text": "hola"}]))
    assert translation.translate_texts(["Hello", "World"]) == ["hola", "mundo"]


def test_duplicate_or_missing_indices_fall_back(monkeypatch):
    monkeypatch.setattr(translation, "_post", lambda payload: reply([{"i": 0, "text": "a"}, {"i": 0, "text": "b"}]))
    monkeypatch.setattr(translation, "local_translate", lambda texts: [f"LOCAL:{t}" for t in texts])
    assert translation.translate_texts(["Hello", "World"]) == ["LOCAL:Hello", "LOCAL:World"]


@pytest.mark.parametrize("bad", [
    {"choices": [{"message": {"content": "no json here"}}]},
    {"choices": [{"message": {"content": json.dumps({"cues": [{"i": 0, "text": ""}]})}}]},
    {"choices": [{"message": {"content": json.dumps({"cues": [{"i": 0, "text": "a"}, {"i": 1, "text": "b"}]})}}]},
    {"choices": [{"message": {"content": json.dumps({"cues": [{"i": 5, "text": "wrong index"}]})}}]},
])
def test_invalid_reply_retries_then_uses_local_model(monkeypatch, bad):
    attempts = []

    def fake_post(payload):
        attempts.append(payload)
        return bad

    monkeypatch.setattr(translation, "_post", fake_post)
    monkeypatch.setattr(translation, "local_translate", lambda texts: ["local" for _ in texts])
    assert translation.translate_texts(["Hello"]) == ["local"]
    assert len(attempts) == 2, "one retry before falling back"


def test_missing_local_model_cannot_fall_back(monkeypatch):
    monkeypatch.setattr(translation, "local_available", lambda: False)
    monkeypatch.setattr(translation, "_post", lambda payload: (_ for _ in ()).throw(RuntimeError("down")))
    with pytest.raises(RuntimeError):
        translation.translate_texts(["Hello"])


def test_no_local_fallback_raises(monkeypatch):
    monkeypatch.setattr(translation, "_post", lambda payload: {"choices": [{"message": {"content": "junk"}}]})
    monkeypatch.setattr(translation, "LOCAL_FALLBACK", False)
    with pytest.raises(ValueError):
        translation.translate_texts(["Hello"])


def test_http_failure_falls_back(monkeypatch):
    def boom(payload):
        raise RuntimeError("translation endpoint returned HTTP 403")

    monkeypatch.setattr(translation, "_post", boom)
    monkeypatch.setattr(translation, "local_translate", lambda texts: ["local" for _ in texts])
    assert translation.translate_texts(["Hello"]) == ["local"]


def test_local_only_when_proxy_unconfigured(monkeypatch):
    monkeypatch.setattr(translation, "ENDPOINT", "")
    monkeypatch.setattr(translation, "_post", lambda payload: pytest.fail("proxy must not be called"))
    monkeypatch.setattr(translation, "local_translate", lambda texts: [f"LOCAL:{t}" for t in texts])
    assert translation.translate_texts(["Hello"]) == ["LOCAL:Hello"]


def test_proxy_content_parts_and_code_fences(monkeypatch):
    body = {"choices": [{"message": {"content": [{"text": '```json\n{"cues":[{"i":0,"text":"Hola"}]}\n```'}]}}]}
    monkeypatch.setattr(translation, "_post", lambda payload: body)
    assert translation.translate_texts(["Hello"]) == ["Hola"]


def test_srt_timings_preserved_through_proxy(monkeypatch):
    def fake_post(payload):
        cues = json.loads(payload["messages"][1]["content"])["cues"]
        return reply([{"i": c["i"], "text": f"ES {c['i']}"} for c in cues])

    monkeypatch.setattr(translation, "_post", fake_post)
    monkeypatch.setattr(translation, "BATCH_CUES", 16)
    data = "\n".join(f"{i}\n00:00:00,100 --> 00:00:00,900\nEnglish {i}\n" for i in range(1, 34)).encode()
    result = translation.translate_srt(data).decode()
    assert result.count("00:00:00,100 --> 00:00:00,900") == 33
    assert "33\n00:00:00,100" in result
    assert "English" not in result


def test_spanish_api_uses_proxy_and_keeps_english_asr(monkeypatch):
    model = FakePipeline()
    monkeypatch.setattr(arc, "pipeline", lambda: model)
    monkeypatch.setattr(translation, "_post", lambda payload: reply([{"i": 0, "text": "Hola mundo"}]))
    response = client.post("/asr?task=translate&language=en&target_language=es", files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert response.status_code == 200
    assert "00:00:00,500 --> 00:00:01,000\nHola mundo" in response.text
    assert model.calls[0][1]["task"] == "transcribe"
    assert model.calls[0][1]["language"] == "<|en|>"


def test_translation_failure_does_not_return_english(monkeypatch):
    monkeypatch.setattr(arc, "pipeline", lambda: FakePipeline())
    monkeypatch.setattr(translation, "_post", lambda payload: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(translation, "LOCAL_FALLBACK", False)
    response = client.post("/asr?task=translate&language=en&target_language=es", files={"audio_file": ("audio", b"\0\0" * 16000)})
    assert response.status_code == 503
    assert "Hello world" not in response.text


@pytest.mark.parametrize("query", ["language=fr&task=translate", "language=en&task=transcribe", "language=en&task=translate&target_language=de"])
def test_invalid_pairs_rejected(monkeypatch, query):
    if "target_language" not in query:
        query += "&target_language=es"
    assert client.post("/asr?" + query, files={"audio_file": ("audio", b"\0\0")}).status_code == 400


def test_empty_srt_passthrough_and_rejects_non_srt(monkeypatch):
    monkeypatch.setattr(translation, "_post", lambda payload: pytest.fail("no cues, no call"))
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
    source = '''class MatchedSubtitle:
    def get_matches(self, video):
        if isinstance(video, Episode):
            self.matches.update(["series", "season", "episode"])
        elif isinstance(video, Movie):
            self.matches.update(["title"])

        return self.matches

class Provider:
    def query(self, language, sub):
        video = sub.video
        if sub.task == "translate":
            if language.alpha3 != "eng":
                return None
        sub.release_info = f'{sub.task} test'
        return sub
    def detect(self, path):
        if True:
            video_name = path if self.pass_video_name else None
    def download(self, subtitle, input_language, video_name, out):
        if subtitle.task == "error":
            return
        if subtitle.task == "transcribe":
            output_language = subtitle.audio_language
        else:
            output_language = "eng"
        video_name = subtitle.video.original_path if self.pass_video_name else None
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
    from types import SimpleNamespace
    env = {"os": __import__("os"), "ConfigurationError": ValueError}
    exec(patched, env)
    monkeypatch.setenv("WHISPER_TRANSLATION_ENDPOINT", "http://whisper-arc:9000")
    provider = env["Provider"]()
    provider.endpoint = "http://subgen:9000"
    sub = SimpleNamespace(task="translate", audio_language="eng",
                          video=SimpleNamespace(original_path="/data/media/series/99/Season 01/Show.S01E02.mkv"))
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is None
    sub.language = SimpleNamespace(alpha3="spa")
    with pytest.raises(ValueError, match="configured Arc endpoint and English audio for /data/media/series/99"):
        provider.download(sub, "en", None, b"pcm")
    provider.endpoint = "http://whisper-arc:9000"
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is sub
    assert "Show.S01E02.mkv" in sub.release_info
    # Metadata boosts are based on the exact input path and known equal fields.
    class Episode:
        pass
    class Movie:
        pass
    env.update(Episode=Episode, Movie=Movie)
    video = Episode()
    video.original_path = sub.video.original_path
    video.year, video.source, video.release_group = 2017, "Blu-ray", "HANDJOB"
    matched = env["MatchedSubtitle"]()
    matched.matches = set()
    matched.arc_source_path = video.original_path
    matched.arc_source_fields = {"year": 2017, "source": "Blu-ray", "release_group": "HANDJOB", "resolution": None}
    assert matched.get_matches(video) == {"series", "season", "episode", "year", "source", "release_group"}
    video.source = "Web"
    assert "source" not in matched.get_matches(video)
    assert "hash" not in matched.matches
    video.original_path = "/different/episode.mkv"
    assert matched.get_matches(video) == set()
    sub.audio_language = "fra"
    assert provider.query(SimpleNamespace(alpha3="spa"), sub) is None
