# Whisper Arc for Bazarr

Local, Bazarr-compatible AI subtitles using **Intel Arc + OpenVINO Whisper**, with an optional **English → Spanish translation layer**. No API keys, cloud inference, media mounts, or published host ports.

- `/asr`: raw 16 kHz mono s16le PCM (`encode=false`) or FFmpeg-readable upload (`encode=true`), SRT output.
- `/detect-language`: Bazarr-compatible `language_code` and `detected_language` JSON.
- `/status`: version for Bazarr's Test Connection button; `/health`: GPU/model availability.
- Whisper base runs on **GPU explicitly**, never silently on CPU.
- Spanish text translation preserves cue numbering and timestamps, and runs on either:
  - an OpenAI-compatible **LiteLLM proxy** (primary; the fleet proxy on `openweb-ui`), or
  - the bundled **OPUS-MT/CTranslate2 int8** local model, used automatically when the proxy is missing or fails.

Tested on an **Intel Arc A380**, including Docker inside Proxmox LXC. The host must expose `/dev/dri` and the LXC must already have permission to access it. This project does not change Proxmox/BIOS settings or install host GPU drivers.

## Start

Requires Docker Compose and Intel GPU drivers on the host. Clone and build:

```bash
git clone https://github.com/bartivs/whisper-arc-bazarr.git
cd whisper-arc-bazarr
docker build -t whisper-arc-bazarr:local .
sudo install -d -o 65534 -g 65534 -m 0750 models/whisper models/translation
```

Download pinned models (network needed for bootstrap only):

```bash
docker run --rm --user 65534:65534 -v "$PWD/models/whisper:/models" \
  whisper-arc-bazarr:local python3 bootstrap.py /models
docker run --rm --user 65534:65534 -v "$PWD/models/translation:/translation-model" \
  whisper-arc-bazarr:local python3 bootstrap.py /translation-model translation
```

Join Bazarr's existing Docker network; replace `servarrnetwork` if needed:

```bash
export BAZARR_NETWORK=servarrnetwork
export RENDER_GID="$(stat -c '%g' /dev/dri/renderD128)"
docker compose up -d --build
```

No port is published. Use `docker exec` or another container on the same network to test. Models are mounted read-only at runtime; files persist on the host. The derivative image extends a pinned, tested Intel runtime from Whisper Pro ASR but **does not run that project's application**; it starts this small FastAPI adapter instead. It adds only the SentencePiece dependency. The base image is several GB; "small" refers to this adapter/model, not container download size.

## Bazarr: transcription and native Whisper translation

Settings → Providers → enable **Whisper**, endpoint **`http://whisper-arc:9000`**. Keep connection timeout 5s; start with read timeout 3600s. Turn off “Pass video filename”: audio is uploaded, and paths are neither required nor read.

Select your language profile and use **Manual Search** on one episode. Whisper supports same-language transcription and native translation **into English**. Without the extension below, Bazarr does not list English-to-Spanish translations. Exclude `whisperai` from automatic audio sync. For automatic downloads, note Whisper's low fixed scores (220/360 episodes, 60/180 movies); lowering global scores can admit bad matches from other providers.

## Bazarr: English → Spanish extension

The built-in Whisper provider filters out Spanish translation results **before** calling the API. The included startup hook enables only English → Spanish, only for an explicitly selected endpoint. Other endpoints (such as CPU Subgen) retain upstream behavior.

The hook is tested against this LinuxServer Bazarr image:

```
lscr.io/linuxserver/bazarr@sha256:5d916d07404296ec35ee726e13e0e558f05952724cf494a7f009d913fb2b12f3
```

Merge into your existing **Bazarr** service, preserving your existing media/config mounts, network, user and ports:

```yaml
services:
  bazarr:
    image: lscr.io/linuxserver/bazarr@sha256:5d916d07404296ec35ee726e13e0e558f05952724cf494a7f009d913fb2b12f3
    environment:
      - WHISPER_TRANSLATION_ENDPOINT=http://whisper-arc:9000
    volumes:
      - ./bazarr/90-whisper-translation.sh:/custom-cont-init.d/90-whisper-translation.sh:ro
      - ./bazarr/patch_provider.py:/opt/whisper-arc/patch_provider.py:ro
```

Ensure `bazarr/90-whisper-translation.sh` is executable, then recreate **only Bazarr** using your own stack command. The hook verifies a known upstream SHA-256 and compiles the result before writing. It is idempotent; unknown provider versions are rejected, not modified. Review/retest the hook before upgrading the Bazarr pin. LinuxServer may continue starting after an init-hook error, so check logs for `Whisper Arc Spanish provider extension verified`; a running container alone does not prove the extension loaded.

In Bazarr keep the Whisper endpoint above, select a **Spanish** language profile, then manually search an English-audio episode. The result appears under **Whisper**, not as a separate provider. Clicking download transcribes on Arc then translates text on CPU. Spanish-audio transcription remains native Whisper.

To roll back: remove both hook mounts and `WHISPER_TRANSLATION_ENDPOINT` and recreate Bazarr from its image. Switch endpoint to your previous ASR/Subgen service for CPU fallback (that fallback cannot translate English → Spanish).

## Translation engine: LiteLLM proxy (recommended) or local model

Spanish cue text is translated by the first engine that is configured and healthy:

1. **LiteLLM / OpenAI-compatible endpoint** (`TRANSLATION_ENDPOINT` + `TRANSLATION_API_KEY`).
2. **Bundled local model** (automatic fallback, or the only engine when no endpoint/key is set).

Only the translation text ever leaves the host, and only when the proxy path is used: your **audio never leaves** the machine, and neither do cue numbers or timestamps. Set an empty `TRANSLATION_API_KEY` for a fully offline deployment.

Point it at any OpenAI-compatible `/chat/completions` endpoint:

```bash
TRANSLATION_ENDPOINT=http://192.168.1.109:4000/v1   # e.g. a LiteLLM proxy
TRANSLATION_MODEL=subtitle-translate                # proxy model alias
TRANSLATION_API_KEY=sk-...                          # scoped virtual key, not a master key
TRANSLATION_LOCAL_FALLBACK=true                     # degrade to the local model on failure
```

To use a **separate LiteLLM instance** (no Docker network needed), add a model alias to its `model_list` and request it by that alias:

```yaml
  - model_name: subtitle-translate
    litellm_params:
      model: openrouter/google/gemini-3.1-flash-lite
      api_key: os.environ/OPENROUTER_API_KEY
```

**Model choice.** An independent 1,002-segment subtitle benchmark across six target languages ranked a Gemini Flash Lite lane second overall and best on quality-to-cost, and found English→Spanish mature with small gaps between models — so the cheap lane is the right pick for this direction. Cost is roughly **$0.02 per feature film** at current OpenRouter pricing (~10k input + ~12k output tokens). Because the model lives behind a proxy alias, changing it needs no change on the Whisper host.

**Use a scoped key.** The adapter only needs one model. Mint a LiteLLM virtual key limited to that alias instead of copying the master key:

```bash
curl -sS -X POST http://<litellm>:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"key_alias":"whisper-arc-translate","models":["subtitle-translate"]}'
```

The returned `sk-...` value is the `TRANSLATION_API_KEY`. Rotate by deleting the old key (`/key/delete`) and repeating. Note the key is sent over plain HTTP unless your endpoint terminates TLS, so keep it on a trusted network.

**Reliability.** Each batch is a numbered JSON request, and a reply is accepted only when it has exactly one non-empty translation per requested cue index; otherwise it is retried once and then handed to the local model. Reordered replies are re-ordered by index rather than discarded. Translation failures never yield English subtitles labelled as Spanish.

## API smoke test

From a container on Bazarr's network, given a short PCM test clip:

```bash
curl -f -F audio_file=@clip.pcm \
  'http://whisper-arc:9000/asr?encode=false&language=en&task=translate&target_language=es&output=srt'
curl -f -F audio_file=@clip.pcm \
  'http://whisper-arc:9000/detect-language?encode=false'
```

Omitting `target_language` preserves native Whisper behavior. Spanish targets require `language=en&task=translate`; unsupported pairs return 400. Translation errors return 503, never English mislabeled as Spanish. An active request rejects concurrent work with 503; do not enable bulk parallel searches.

## Tests

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-test.txt
# ffmpeg must be installed for the WAV test
pytest -q
```

Unit/API tests use fake inference so they run without GPU or model weights. They validate chunk timestamps, detection, target validation, translation error handling, cue alignment, and hook compatibility/scoping. Real GPU/provider smoke tests must also be run on the deployment host.

## Limitations

- Experimental: manually inspect subtitles before using across a library.
- Whisper `base` is fast but less accurate than larger models; hallucinations and speech split at 30-second window boundaries are possible.
- Translation is batched per cue group, not whole-film context; quality and line-length conventions are not guaranteed. Review samples before bulk use.
- With the proxy engine enabled, subtitle **text** is sent to the configured provider (e.g. OpenRouter) and is subject to that provider's retention policy. Audio never leaves the host. Disable by clearing `TRANSLATION_API_KEY` to stay fully offline.
- English → standard Spanish only in the text layer. Other requested targets fail explicitly.
- One inference request at a time, 2 CPU / 2 GiB cap in Compose. First GPU initialization is slower. The adapter retains its models in memory.
- The unauthenticated API belongs on a trusted private Docker network. Do not expose it publicly. Uploads over 512 MiB are rejected by application logic after multipart parsing; use a reverse proxy body-size limit if accepting untrusted clients.
- Uploaded audio uses temporary disk files (removed after each request); no media paths are followed and no subtitles are written into your library by the service. Bazarr performs the final save.

## Attribution and licensing

Adapter and hook code: MIT (see LICENSE). No Bazarr source or model weights are bundled in this repository.

- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) and [OpenVINO Whisper base model](https://huggingface.co/OpenVINO/whisper-base-fp16-ov), model revision pinned in `src/bootstrap.py`.
- [OpenAI Whisper](https://github.com/openai/whisper), the underlying ASR model; consult upstream/model licenses.
- [Helsinki-NLP OPUS-MT en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es), **CC BY 4.0**, using [michaelfeil's CTranslate2 conversion](https://huggingface.co/michaelfeil/ct2fast-opus-mt-en-es), pinned revision. Credit Helsinki-NLP/OPUS-MT and the conversion author when redistributing weights. This adapter does not alter model weights.
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) and [SentencePiece](https://github.com/google/sentencepiece).
- Pinned Intel runtime base from [Whisper Pro ASR](https://github.com/ventura8/Whisper-Pro-ASR); upstream application is bypassed, its container/dependency licenses still apply.
- [Bazarr](https://github.com/morpheus65535/bazarr) and [LinuxServer Bazarr container](https://docs.linuxserver.io/images/docker-bazarr/); runtime extension patches operator-installed code. Not affiliated with those projects.
- Translation via a [LiteLLM](https://github.com/BerriAI/litellm) proxy and OpenRouter is **optional** and not bundled. Model choice, availability and pricing belong to those services.

Ansible integration: [bartivs/homelab-infra](https://github.com/bartivs/homelab-infra), role `media-stack`.
