"""Language settings in Unix/POSIX notation plus a source-language classifier.

Source evidence is considered in priority order: a filename language hint,
Whisper's acoustic detection, then a lexical check of the transcript.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass

_PATTERN = re.compile(r"^([a-zA-Z]{2})(?:[_-]([a-zA-Z0-9]{2,3}))?(?:\.(?:UTF-?8|utf-?8))?$")
_REGIONS = {"MX": "Mexico", "ES": "Spain", "AR": "Argentina", "US": "United States",
            "GB": "United Kingdom", "419": "Latin America", "FR": "France",
            "DE": "Germany", "BR": "Brazil", "PT": "Portugal", "CA": "Canada",
            "CN": "China", "TW": "Taiwan", "JP": "Japan", "KR": "South Korea"}
# ISO 639-1 names are used in prompts, not arbitrary user-supplied instructions.
_LANGUAGE_NAMES = dict(item.split("=", 1) for item in (
    "aa=Afar;ab=Abkhazian;ae=Avestan;af=Afrikaans;ak=Akan;am=Amharic;an=Aragonese;ar=Arabic;"
    "as=Assamese;av=Avaric;ay=Aymara;az=Azerbaijani;ba=Bashkir;be=Belarusian;bg=Bulgarian;"
    "bh=Bihari;bi=Bislama;bm=Bambara;bn=Bengali;bo=Tibetan;br=Breton;bs=Bosnian;ca=Catalan;"
    "ce=Chechen;ch=Chamorro;co=Corsican;cr=Cree;cs=Czech;cu=Church Slavonic;cv=Chuvash;cy=Welsh;"
    "da=Danish;de=German;dv=Divehi;dz=Dzongkha;ee=Ewe;el=Greek;en=English;eo=Esperanto;es=Spanish;"
    "et=Estonian;eu=Basque;fa=Persian;ff=Fulah;fi=Finnish;fj=Fijian;fo=Faroese;fr=French;fy=Frisian;"
    "ga=Irish;gd=Gaelic;gl=Galician;gn=Guarani;gu=Gujarati;gv=Manx;ha=Hausa;he=Hebrew;hi=Hindi;"
    "ho=Hiri Motu;hr=Croatian;ht=Haitian Creole;hu=Hungarian;hy=Armenian;hz=Herero;ia=Interlingua;"
    "id=Indonesian;ie=Interlingue;ig=Igbo;ii=Sichuan Yi;ik=Inupiaq;io=Ido;is=Icelandic;it=Italian;"
    "iu=Inuktitut;ja=Japanese;jv=Javanese;ka=Georgian;kg=Kongo;ki=Kikuyu;kj=Kuanyama;kk=Kazakh;"
    "kl=Kalaallisut;km=Khmer;kn=Kannada;ko=Korean;kr=Kanuri;ks=Kashmiri;ku=Kurdish;kv=Komi;"
    "kw=Cornish;ky=Kirghiz;la=Latin;lb=Luxembourgish;lg=Ganda;li=Limburgan;ln=Lingala;lo=Lao;"
    "lt=Lithuanian;lu=Luba-Katanga;lv=Latvian;mg=Malagasy;mh=Marshallese;mi=Maori;mk=Macedonian;"
    "ml=Malayalam;mn=Mongolian;mr=Marathi;ms=Malay;mt=Maltese;my=Burmese;na=Nauru;nb=Norwegian Bokmal;"
    "nd=North Ndebele;ne=Nepali;ng=Ndonga;nl=Dutch;nn=Norwegian Nynorsk;no=Norwegian;nr=South Ndebele;"
    "nv=Navajo;ny=Chichewa;oc=Occitan;oj=Ojibwa;om=Oromo;or=Oriya;os=Ossetian;pa=Punjabi;pi=Pali;"
    "pl=Polish;ps=Pashto;pt=Portuguese;qu=Quechua;rm=Romansh;rn=Rundi;ro=Romanian;ru=Russian;"
    "rw=Kinyarwanda;sa=Sanskrit;sc=Sardinian;sd=Sindhi;se=Northern Sami;sg=Sango;si=Sinhala;sk=Slovak;"
    "sl=Slovenian;sm=Samoan;sn=Shona;so=Somali;sq=Albanian;sr=Serbian;ss=Swati;st=Southern Sotho;"
    "su=Sundanese;sv=Swedish;sw=Swahili;ta=Tamil;te=Telugu;tg=Tajik;th=Thai;ti=Tigrinya;tk=Turkmen;"
    "tl=Tagalog;tn=Tswana;to=Tonga;tr=Turkish;ts=Tsonga;tt=Tatar;tw=Twi;ty=Tahitian;ug=Uighur;"
    "uk=Ukrainian;ur=Urdu;uz=Uzbek;ve=Venda;vi=Vietnamese;vo=Volapuk;wa=Walloon;wo=Wolof;xh=Xhosa;"
    "yi=Yiddish;yo=Yoruba;za=Zhuang;zh=Chinese;zu=Zulu"
).split(";"))
_REGION_CODES = set((
    "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ "
    "CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR "
    "GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP "
    "KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ "
    "NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ "
    "TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW 419"
).split())
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
        name = _LANGUAGE_NAMES[self.language]
        return name + (f" ({_REGIONS.get(self.region, self.region)})" if self.region else "")


def parse_locale(value):
    match = _PATTERN.fullmatch(value.strip())
    if not match:
        raise ValueError("Use a supported language code or Unix locale, e.g. es_MX.UTF-8")
    language, region = match.groups()
    language, region = language.lower(), (region or "").upper()
    if language not in _LANGUAGE_NAMES or (region and region not in _REGION_CODES):
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
    """Priority chain: filename hint > acoustic detection > lexical transcript check.

    Returns (code, evidence) where evidence names the deciding layer, so the
    choice is auditable in logs instead of being silently assumed.
    """
    tag = (container_tag or "").strip().lower()
    acoustic = (whisper_code or "").strip().lower()
    # Whisper's legacy code for Javanese differs from ISO 639-1.
    tag = "jv" if tag == "jw" else tag
    acoustic = "jv" if acoustic == "jw" else acoustic
    lexical, lexical_confidence = classify_text(text)
    for code, source in ((tag, "container_tag"), (acoustic, "whisper_acoustic")):
        if code in _LANGUAGE_NAMES:
            # Lexical evidence never overrides a usable higher-priority hint.
            return code, source
    if lexical:
        return lexical, f"text_classifier:{lexical_confidence}"
    return fallback, "fallback"


SOURCE_LOCALE = parse_locale(os.environ.get("TRANSLATION_SOURCE_LOCALE", "en_US.UTF-8"))
TARGET_LOCALE = parse_locale(os.environ.get("TRANSLATION_TARGET_LOCALE", "es_MX.UTF-8"))


def target_locale(value):
    """A bare configured language inherits its region; other targets stand alone."""
    locale = parse_locale(value)
    if not locale.region and locale.language == TARGET_LOCALE.language:
        return TARGET_LOCALE
    return locale
