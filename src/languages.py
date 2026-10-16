"""Language settings in Unix/POSIX notation, without changing the OS locale."""

import os
import re
from dataclasses import dataclass

_PATTERN = re.compile(r"^([a-zA-Z]{2})(?:[_-]([a-zA-Z]{2}))?(?:\.(?:UTF-?8|utf-?8))?$")
_REGIONS = {"MX": "Mexico", "ES": "Spain", "AR": "Argentina", "US": "United States", "GB": "United Kingdom"}


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
    allowed = {"en": {"", "US", "GB"}, "es": {"", "MX", "ES", "AR", "US"}}
    if language not in allowed or region not in allowed[language]:
        raise ValueError("Unsupported language/region combination")
    return Locale(language, region)


SOURCE_LOCALE = parse_locale(os.environ.get("TRANSLATION_SOURCE_LOCALE", "en_US.UTF-8"))
TARGET_LOCALE = parse_locale(os.environ.get("TRANSLATION_TARGET_LOCALE", "es_MX.UTF-8"))
if SOURCE_LOCALE.language != "en" or TARGET_LOCALE.language != "es":
    raise ValueError("The configured text translation pair must be English to Spanish")
