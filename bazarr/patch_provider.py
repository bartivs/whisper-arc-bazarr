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
MARKER = "# whisper-arc-bazarr multilingual provider extension v5"
PREVIOUS_MARKERS = tuple(f"# whisper-arc-bazarr Spanish provider extension v{n}" for n in (1, 2, 3, 4))
REPLACEMENTS = [
    ('            if language.alpha3 != "eng":',
     # Keep upstream English translation elsewhere; Arc accepts ISO 639-1 targets.
     '            if language.alpha3 != "eng" and not (len(wlm.alpha3_to_alpha2(language.alpha3) or "") == 2 and self.endpoint == os.environ.get("WHISPER_TRANSLATION_ENDPOINT")):'),
    ('        if subtitle.task == "error":',
     '        if subtitle.task == "translate" and subtitle.language.alpha3 != "eng" and self.endpoint != os.environ.get("WHISPER_TRANSLATION_ENDPOINT"):\n'
     '            # Name the file so a refused download can be found in the Bazarr log.\n'
     '            raise ConfigurationError(f"Text translation requires the configured Arc endpoint for {subtitle.video.original_path}")\n'
     '        if subtitle.task == "error":'),
    ('            output_language = "eng"',
     '            output_language = subtitle.language.alpha3'),
    ("                                      'video_file': video_name},\n                              files={'audio_file': out},",
     "                                      'video_file': video_name,\n"
     "                                      **({'target_language': subtitle.language.alpha2} if subtitle.task == 'translate' and output_language != 'eng' and self.endpoint == os.environ.get('WHISPER_TRANSLATION_ENDPOINT') else {})},\n"
     "                              files={'audio_file': out},"),
    ("        subtitle.content = r.content", "        r.raise_for_status()\n        subtitle.content = r.content"),
    ('''    def get_matches(self, video):
        if isinstance(video, Episode):
            self.matches.update(["series", "season", "episode"])
        elif isinstance(video, Movie):
            self.matches.update(["title"])

        return self.matches
''', '''    def get_matches(self, video):
        bound = getattr(self, "arc_source_path", None)
        if bound is not None:
            self.matches.clear()
            if os.path.normpath(video.original_path) != bound:
                return self.matches
        if isinstance(video, Episode):
            self.matches.update(["series", "season", "episode"])
        elif isinstance(video, Movie):
            self.matches.update(["title"])
        if bound is not None:
            # Generated from this exact input, not inferred from an external release.
            # Capped on purpose: 220 base + year(90) + release_group(20) = 330/360
            # (92%). That clears Bazarr's minimum_score (90) so a generated subtitle
            # is still downloaded when nothing better exists, yet stays below the
            # ~93% a real release usually scores, so a genuine download is picked
            # first. This is a matching score, not a quality claim.
            self.matches.update(key for key, value in self.arc_source_fields.items()
                                if value is not None and value != "" and value == getattr(video, key, None))
        return self.matches
'''),
    ("        sub.release_info = f'{sub.task}",
     '        if self.endpoint == os.environ.get("WHISPER_TRANSLATION_ENDPOINT"):\n'
     '            sub.arc_source_path = os.path.normpath(video.original_path)\n'
     '            sub.arc_source_fields = {key: getattr(video, key, None) for key in\n'
     '                ("year", "release_group")}\n'
     "        sub.release_info = f'{os.path.basename(video.original_path) + \" | \" if getattr(sub, \"arc_source_path\", None) else \"\"}{sub.task}"),
    ('            video_name = path if self.pass_video_name else None',
     '            video_name = path if self.pass_video_name or self.endpoint == os.environ.get("WHISPER_TRANSLATION_ENDPOINT") else None'),
    ('        video_name = subtitle.video.original_path if self.pass_video_name else None',
     '        video_name = subtitle.video.original_path if self.pass_video_name or self.endpoint == os.environ.get("WHISPER_TRANSLATION_ENDPOINT") else None'),
]


def patch(source):
    if any(source.endswith("\n" + marker + "\n") for marker in PREVIOUS_MARKERS):
        # Only an in-place container layer can hold the old marker; a recreated
        # container starts from the pristine image file and takes the patched path.
        raise RuntimeError("Older Whisper extension found; recreate the Bazarr container to re-apply this hook")
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
    print("Whisper Arc multilingual provider extension verified")


if __name__ == "__main__":
    main()
