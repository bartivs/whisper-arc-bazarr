"""Scoped extension for the pinned Bazarr Whisper provider; no upstream code vendored.

Run from LinuxServer's custom-cont-init.d hook. Refuse unknown source, compile
before replacing, and accept repeat runs only when the complete patched source
hash matches. Remove hook and recreate Bazarr to restore upstream code.
"""

import hashlib
import os
import sys
from pathlib import Path

UPSTREAM_SHA256 = "257c450de9c0e2fa86a30582f9b8b4446756b197755c2c354a9ed604bdc92ed7"  # pragma: allowlist secret (public source checksum)
MARKER = "# whisper-arc-bazarr Spanish provider extension v1"
REPLACEMENTS = [
    ('            if language.alpha3 != "eng":',
     '            if language.alpha3 != "eng" and not (language.alpha3 == "spa" and sub.audio_language == "eng" and self.endpoint == os.environ.get("WHISPER_TRANSLATION_ENDPOINT")):'),
    ('        if subtitle.task == "error":',
     '        if subtitle.task == "translate" and subtitle.language.alpha3 == "spa" and (subtitle.audio_language != "eng" or self.endpoint != os.environ.get("WHISPER_TRANSLATION_ENDPOINT")):\n'
     '            raise ConfigurationError("Spanish translation requires the configured Arc endpoint and English audio")\n'
     '        if subtitle.task == "error":'),
    ('            output_language = "eng"',
     '            output_language = subtitle.language.alpha3'),
    ("                                      'video_file': video_name},\n                              files={'audio_file': out},",
     "                                      'video_file': video_name,\n"
     "                                      **({'target_language': 'es'} if output_language == 'spa' and self.endpoint == os.environ.get('WHISPER_TRANSLATION_ENDPOINT') else {})},\n"
     "                              files={'audio_file': out},"),
    ("        subtitle.content = r.content", "        r.raise_for_status()\n        subtitle.content = r.content"),
]


def patch(source):
    if source.endswith("\n" + MARKER + "\n"):
        # Reverse only exact changes and re-verify the original, rather than
        # trusting a marker an incompatible or partially patched file could retain.
        original = source.removesuffix("\n" + MARKER + "\n")
        for before, after in reversed(REPLACEMENTS):
            if original.count(after) != 1:
                raise RuntimeError("Incomplete existing Whisper extension")
            original = original.replace(after, before)
        if hashlib.sha256(original.encode()).hexdigest() != UPSTREAM_SHA256:
            raise RuntimeError("Unknown already-patched Whisper provider")
        return source
    if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Unsupported Bazarr Whisper provider; review before upgrading")
    for before, after in REPLACEMENTS:
        if source.count(before) != 1:
            raise RuntimeError("Whisper provider patch target is ambiguous")
        source = source.replace(before, after)
    source += "\n" + MARKER + "\n"
    compile(source, "whisperai.py", "exec")
    return source


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/bazarr/bin/custom_libs/subliminal_patch/providers/whisperai.py")
    original = path.read_text()
    result = patch(original)
    if result != original:
        # Preserve owner, mode and inode (LinuxServer may have initialized permissions).
        path.write_text(result)
    print("Whisper Arc Spanish provider extension verified")


if __name__ == "__main__":
    main()
