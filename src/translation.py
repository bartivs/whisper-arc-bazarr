"""Offline English -> Spanish cue-text translation; no model execution of input."""

import os
import re
from pathlib import Path

MODEL_DIR = Path(os.environ.get("TRANSLATION_MODEL_DIR", "/translation-model"))
_translator = None
_source = None
_target = None
TIMING = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")


def translate_texts(texts):
    global _translator, _source, _target
    if _translator is None:
        import ctranslate2
        import sentencepiece as spm

        source = spm.SentencePieceProcessor(model_file=str(MODEL_DIR / "source.spm"))
        target = spm.SentencePieceProcessor(model_file=str(MODEL_DIR / "target.spm"))
        translator = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8", inter_threads=1, intra_threads=2)
        _source, _target, _translator = source, target, translator
    tokens = [_source.encode(text, out_type=str) + ["</s>"] for text in texts]
    results = _translator.translate_batch(tokens, beam_size=4, max_batch_size=16, max_input_length=512, max_decoding_length=512)
    return [_target.decode(result.hypotheses[0]).strip() for result in results]


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
    for start in range(0, len(cues), 16):
        batch = cues[start:start + 16]
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
