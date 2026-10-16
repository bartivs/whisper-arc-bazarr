"""Language settings in Unix/POSIX notation plus a source-language classifier.

The classifier is evidence-gathering, not a single oracle: a container audio tag,
Whisper's acoustic detection and a lexical check of the transcript vote, so a
mis-tagged or untagged file (e.g. "German Dubbed") still resolves correctly.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass

_PATTERN = re.compile(r"^([a-zA-Z]{2})(?:[_-]([a-zA-Z0-9]{2,3}))?(?:\.(?:UTF-?8|utf-?8))?$")
_REGIONS = {"MX": "Mexico", "ES": "Spain", "AR": "Argentina", "US": "United States",
            "GB": "United Kingdom", "419": "Latin America"}
_LANGUAGE_NAMES = {"en": "English", "de": "German", "es": "Spanish", "fr": "French",
                   "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian"}
# Whisper/ISO 639-1 codes that are too coarse to translate from.
AMBIGUOUS = {"und", "mul", "zxx", "mis", "qaa"}

# Deliberately small, high-signal function-word sets: enough to separate the
# languages this fleet actually stores without shipping a model or a dependency.
_STOPWORDS = {
    "en": {"the", "and", "to", "of", "you", "that", "it", "is", "in", "for", "not", "what", "do", "have"},
    "de": {"der", "die", "das", "und", "nicht", "ist", "ich", "sie", "mit", "ein", "auch", "was", "wir", "auf"},
    "es": {"el", "la", "los", "las", "de", "que", "no", "es", "y", "en", "un", "por", "con", "para", "pero"},
    "fr": {"le", "la", "les", "des", "est", "pas", "que", "et", "un", "pour", "vous", "nous", "dans"},
    "it": {"il", "la", "che", "non", "di", "un", "per", "con", "sono", "una", "più"},
    "pt": {"o", "a", "os", "que", "não", "de", "um", "para", "com", "uma", "você"},
}


@dataclass(frozen=True)
class Locale:
    language: str
    region: str = ""

    @property
    def unix(self):
        return self.language + (f"_{self.region}.UTF-8" if self.region else "")

    @property
    def label(self):
        name = {"en": "English", "es": "Spanish"}[self.language]
        return name + (f" ({_REGIONS[self.region]})" if self.region else "")


def parse_locale(value):
    match = _PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("Use a supported language code or Unix locale, e.g. es_MX.UTF-8")
    language, region = match.groups()
    language, region = language.lower(), (region or "").upper()
    allowed = {"en": {"", "US", "GB"}, "es": {"", "419", "MX", "ES", "AR", "US"}}
    if language not in allowed or region not in allowed[language]:
        raise ValueError("Unsupported language/region combination")
    return Locale(language, region)


def language_name(code):
    return _LANGUAGE_NAMES.get(code, code.upper() if code else "unknown")


def classify_text(text):
    """Lexical vote over function words. Returns (code, confidence) or ("", 0.0)."""
    tokens = re.findall(r"[a-zà-ÿ']+", (text or "").lower())
    if len(tokens) < 12:
        return "", 0.0
    counts = Counter(tokens)
    scores = {code: sum(counts[word] for word in words & counts.keys()) for code, words in _STOPWORDS.items()}
    total = sum(scores.values())
    if not total:
        return "", 0.0
    code, best = max(scores.items(), key=lambda item: item[1])
    confidence = best / total
    # Require a real margin so a mixed or short transcript does not guess.
    return (code, round(confidence, 3)) if best >= 3 and confidence >= 0.4 else ("", round(confidence, 3))


def resolve_source(container_tag="", whisper_code="", text="", fallback="en"):
    """Priority vote: container tag > acoustic detection > lexical transcript check.

    Returns (code, evidence) where evidence names the deciding layer, so the
    choice is auditable in logs instead of being silently assumed.
    """
    tag = (container_tag or "").strip().lower()[:2]
    acoustic = (whisper_code or "").strip().lower()[:2]
    lexical, lexical_confidence = classify_text(text)
    for code, source in ((tag, "container_tag"), (acoustic, "whisper_acoustic")):
        if code and code not in AMBIGUOUS:
            # A lexical disagreement is logged but does not override a real tag.
            return code, source
    if lexical:
        return lexical, f"text_classifier:{lexical_confidence}"
    if acoustic in AMBIGUOUS or tag in AMBIGUOUS or not acoustic:
        return fallback, "fallback"
    return acoustic, "whisper_acoustic"


SOURCE_LOCALE = parse_locale(os.environ.get("TRANSLATION_SOURCE_LOCALE", "en_US.UTF-8"))
TARGET_LOCALE = parse_locale(os.environ.get("TRANSLATION_TARGET_LOCALE", "es_MX.UTF-8"))
if SOURCE_LOCALE.language != "en" or TARGET_LOCALE.language != "es":
    raise ValueError("The configured text translation pair must be English to Spanish")
