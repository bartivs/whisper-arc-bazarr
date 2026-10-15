"""Cue-text translation: fleet LiteLLM proxy first, bundled local model as fallback.

Timings and cue numbers never reach a model. Every batch is checked for length
and index alignment before its text is accepted, so a confused model cannot
silently shift, merge or drop subtitles.
"""

import json
import os
import re
from pathlib import Path
from urllib import error, request

ENDPOINT = os.environ.get("TRANSLATION_ENDPOINT", "").rstrip("/")
MODEL = os.environ.get("TRANSLATION_MODEL", "subtitle-translate")
API_KEY = os.environ.get("TRANSLATION_API_KEY", "")
BATCH_CUES = max(1, int(os.environ.get("TRANSLATION_BATCH_CUES", "40")))
TIMEOUT = float(os.environ.get("TRANSLATION_TIMEOUT", "180"))
TARGET_LANGUAGE = os.environ.get("TRANSLATION_TARGET_LANGUAGE", "Spanish")
LOCAL_MODEL_DIR = Path(os.environ.get("TRANSLATION_MODEL_DIR", "/translation-model"))
LOCAL_FALLBACK = (os.environ.get("TRANSLATION_LOCAL_FALLBACK", "true").strip().lower() not in ("0", "false", "no"))

TIMING = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")

PROMPT = (
    "Translate each English subtitle cue into {language}. Use natural, idiomatic "
    "spoken language and keep the original meaning, tone and line-level brevity. "
    "Never merge, split, reorder, renumber or omit cues. Reply with JSON only in "
    'exactly this shape: {{"cues":[{{"i":<same integer>,"text":"<translation>"}}]}}'
)

_translator = None
_source = None
_target = None


def proxy_configured():
    return bool(ENDPOINT and API_KEY)


def local_available():
    return (LOCAL_MODEL_DIR / "model.bin").is_file()


def _post(payload):
    """Single OpenAI-compatible chat completion against the translation endpoint."""
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{ENDPOINT}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"translation endpoint returned HTTP {exc.code}") from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"translation endpoint unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("translation endpoint returned invalid JSON") from exc


def _extract_cues(body, expected):
    """Pull a validated, index-aligned translation list out of a model reply."""
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("missing completion content") from exc
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise ValueError("non-text completion content")

    raw = _FENCE.sub("", content.strip())
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("model did not return JSON") from exc
    cues = parsed.get("cues") if isinstance(parsed, dict) else None
    if not isinstance(cues, list) or len(cues) != len(expected):
        raise ValueError("model returned the wrong cue count")

    ordered = {}
    for cue in cues:
        if not isinstance(cue, dict) or set(cue) < {"i", "text"} or not isinstance(cue["text"], str):
            raise ValueError("malformed cue in model reply")
        try:
            index = int(cue["i"])
        except (TypeError, ValueError) as exc:
            raise ValueError("non-integer cue index") from exc
        text = " ".join(cue["text"].split())
        if not text:
            raise ValueError("empty translated cue")
        if index in ordered:
            raise ValueError("duplicate cue index")
        ordered[index] = text

    if sorted(ordered) != list(range(len(expected))):
        raise ValueError("model returned unexpected cue indices")
    return [ordered[index] for index in range(len(expected))]


def proxy_translate(texts, attempt=1):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT.format(language=TARGET_LANGUAGE)},
            {"role": "user", "content": json.dumps({"cues": [{"i": i, "text": t} for i, t in enumerate(texts)]}, ensure_ascii=False)},
        ],
    }
    if attempt > 1:
        payload["messages"][0]["content"] += " Your previous reply was unusable: return valid JSON with every cue index."
    body = _post(payload)
    try:
        return _extract_cues(body, texts)
    except ValueError:
        if attempt == 1:
            return proxy_translate(texts, attempt=2)
        raise


def local_translate(texts):
    global _translator, _source, _target
    if _translator is None:
        import ctranslate2
        import sentencepiece as spm

        source = spm.SentencePieceProcessor(model_file=str(LOCAL_MODEL_DIR / "source.spm"))
        target = spm.SentencePieceProcessor(model_file=str(LOCAL_MODEL_DIR / "target.spm"))
        translator = ctranslate2.Translator(str(LOCAL_MODEL_DIR), device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
        _source, _target, _translator = source, target, translator
    tokens = [_source.encode(text, out_type=str) + ["</s>"] for text in texts]
    results = _translator.translate_batch(tokens, beam_size=4, max_batch_size=16, max_input_length=512, max_decoding_length=512)
    return [_target.decode(result.hypotheses[0]).strip() for result in results]


def translate_texts(texts):
    """Translate cues with the proxy when configured, degrading to the local model."""
    if proxy_configured():
        try:
            return proxy_translate(texts)
        except (RuntimeError, ValueError) as exc:
            if not (LOCAL_FALLBACK and local_available()):
                raise
            print(f"translation proxy failed ({exc}); using local model", flush=True)
    return local_translate(texts)


def translate_srt(data):
    """Validate before/after translation. Never give timings/indices to the model."""
    if not data.strip():
        return data
    cues = []
    for block in data.decode("utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3 or not lines[0].isdigit() or not TIMING.fullmatch(lines[1]):
            raise ValueError("Invalid generated SRT cue")
        cues.append((lines[0], lines[1], " ".join(lines[2:])))
    output = []
    for start in range(0, len(cues), BATCH_CUES):
        batch = cues[start:start + BATCH_CUES]
        translated = translate_texts([cue[2] for cue in batch])
        if len(translated) != len(batch):
            raise ValueError("Translation cue count mismatch")
        for (index, timing, _), text in zip(batch, translated):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty translated cue")
            # Prevent model output from injecting additional SRT blocks.
            text = " ".join(text.split())
            output.append(f"{index}\n{timing}\n{text}\n")
    return "\n".join(output).encode("utf-8")
