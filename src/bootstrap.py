"""Download a pinned, small OpenVINO Whisper model once, before starting the service."""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "OpenVINO/whisper-base-fp16-ov"
REVISION = "84fbe975a79a8c996fd32c036558f29e2db6670f"  # pragma: allowlist secret (public model revision)


TRANSLATION_ID = "michaelfeil/ct2fast-opus-mt-en-es"
TRANSLATION_REVISION = "76ec296588e2234f9b7dfad5254219a0f5ecb7af"  # pragma: allowlist secret (public model revision)


def main():
    directory = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "translation":
        snapshot_download(repo_id=TRANSLATION_ID, revision=TRANSLATION_REVISION, local_dir=directory, max_workers=2)
        for name in ("model.bin", "config.json", "source.spm", "target.spm", "shared_vocabulary.txt"):
            if not (directory / name).is_file() or not (directory / name).stat().st_size:
                raise RuntimeError(f"Incomplete translation model: {name}")
        (directory / f".ready-{TRANSLATION_REVISION}").touch()
        return
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=directory, max_workers=2)
    required = {
        "openvino_encoder_model.bin": 30_000_000,
        "openvino_decoder_model.bin": 90_000_000,
        "openvino_encoder_model.xml": 1,
        "openvino_decoder_model.xml": 1,
        "openvino_tokenizer.bin": 1,
        "openvino_detokenizer.bin": 1,
        "generation_config.json": 1,
    }
    for name, minimum in required.items():
        if not (directory / name).is_file() or (directory / name).stat().st_size < minimum:
            raise RuntimeError(f"Incomplete OpenVINO model: {name}")
    (directory / f".ready-{REVISION}").touch()


if __name__ == "__main__":
    main()
