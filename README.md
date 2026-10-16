# Whisper Arc for Bazarr — Intel Arc AI Subtitles

[![Tests](https://github.com/bartivs/whisper-arc-bazarr/actions/workflows/test.yml/badge.svg)](https://github.com/bartivs/whisper-arc-bazarr/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Generate **local AI subtitles for [Bazarr](https://www.bazarr.media/)** with **Intel Arc GPU acceleration**, **OpenAI Whisper** and **OpenVINO**. This self-hosted Docker service turns movie and TV audio into SRT subtitles in three modes:

- **Same-language transcription:** generate subtitles in the original audio language, for languages supported by Whisper.
- **Translation into English:** use Whisper's native translation to turn supported non-English audio into English subtitles, entirely locally.
- **Optional translation into Spanish:** add the Spanish extension for English → Spanish offline, or other source languages → Spanish through a LiteLLM/OpenAI-compatible proxy. The proxy supports a neutral Latin American Spanish prompt.

Spanish is the only target of the **added text-translation layer**, not the only supported subtitle language. Transcription and native English translation need no API keys; only proxy-based Spanish translation requires a configured endpoint/key. No media mounts or published host ports. Audio stays local even when proxy translation is enabled.

## Features

- **Intel Arc speech recognition:** Whisper `base` runs on the GPU through OpenVINO, with no silent CPU fallback.
- **Bazarr Whisper provider compatibility:** use the existing provider for transcription and native translation into English; add the scoped hook for Spanish translations.
- **Offline English → Spanish subtitles:** bundled OPUS-MT/CTranslate2 int8 text translation on CPU.
- **Multilingual → Spanish via LiteLLM:** resolve the source language and translate cue text while preserving SRT timestamps.
- **Self-hosted Docker deployment:** private network, read-only model mounts and a bounded inference queue.

## API overview

- `/asr`: raw 16 kHz mono s16le PCM (`encode=false`) or FFmpeg-readable upload (`encode=true`), SRT output.
- `/detect-language`: Bazarr-compatible `language_code` and `detected_language` JSON.
- `/status`: version for Bazarr's Test Connection button; `/health`: GPU/model availability.
- Whisper base runs on **GPU explicitly**, never silently on CPU.
- Spanish text translation preserves cue numbering and timestamps, and runs on either:
  - an OpenAI-compatible **LiteLLM proxy** (primary; the fleet proxy on `openweb-ui`), or
  - the bundled **OPUS-MT/CTranslate2 int8** local model, used automatically for **English sources only** when the proxy is missing or fails.

Tested on an **Intel Arc A380**, including Docker inside Proxmox LXC. The host must expose `/dev/dri` and the LXC must already have permission to access it. This project does not change Proxmox/BIOS settings or install host GPU drivers.

## Documentation

- [Docker Compose quick start](#quick-start-docker-compose-on-intel-arc)
- [Bazarr Whisper provider setup](#bazarr-transcription-and-native-whisper-translation)
- [Spanish translation extension](#bazarr-multilingual--spanish-extension)
- [LiteLLM and offline translation](#translation-engine-litellm-proxy-recommended-or-local-model)
- [Subtitle matching and scores](#matching-and-scores)
- [Troubleshooting downloads](#diagnosing-failed-downloads)
- [Frequently asked questions](#frequently-asked-questions)
- [Tests](#tests) · [Limitations](#limitations) · [Licensing](#attribution-and-licensing)

## Quick start: Docker Compose on Intel Arc

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

Settings → Providers → enable **Whisper**, endpoint **`http://whisper-arc:9000`**. Keep connection timeout 5s; set the read timeout above the queue wait plus one transcription (3600s is a safe start). Turn on “Pass video filename” for upstream Whisper; the extension below always passes it for the configured Arc endpoint. The path is never read or followed, but its filename supplies language hints and makes a failing episode findable in logs (see [Diagnosing failed downloads](#diagnosing-failed-downloads)).

Select your language profile and use **Manual Search** on one episode. Whisper supports same-language transcription and native translation **into English**. Without the extension below, Bazarr does not list translations into Spanish. Exclude `whisperai` from automatic audio sync. For automatic downloads, note Whisper's low fixed scores (220/360 episodes, 60/180 movies); lowering global scores can admit bad matches from other providers.

## Bazarr: multilingual → Spanish extension

The built-in Whisper provider filters out Spanish translation results **before** calling the API. The included startup hook allows Spanish translations from any source language, only for an explicitly selected endpoint. The adapter resolves the source language before translation; non-English sources need the proxy engine. Other endpoints (such as CPU Subgen) retain upstream behavior.

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

Ensure `bazarr/90-whisper-translation.sh` is executable, then recreate **only Bazarr** using your own stack command. The hook verifies a known upstream SHA-256 and compiles the result before writing. It is idempotent; unknown provider versions are rejected, not modified. The hook runs on every container start, so upgrading this file changes its marker: **recreate** the Bazarr container rather than restarting it, otherwise the hook stops with `Older Whisper extension found; recreate the Bazarr container to re-apply this hook` and the previous patch (still functional) stays in the container layer. Review/retest the hook before upgrading the Bazarr pin. LinuxServer may continue starting after an init-hook error, so check logs for `Whisper Arc Spanish provider extension verified`; a running container alone does not prove the extension loaded.

In Bazarr keep the Whisper endpoint above, select a **Spanish** language profile, then manually search an episode. The result appears under **Whisper**, not as a separate provider. Clicking download transcribes on Arc, then translates cue text through the proxy or, for English sources, the local CPU model. Spanish-audio transcription remains native Whisper.

To roll back: remove both hook mounts and `WHISPER_TRANSLATION_ENDPOINT` and recreate Bazarr from its image. Switch endpoint to your previous ASR/Subgen service for CPU fallback (that fallback cannot translate English → Spanish).

## Translation engine: LiteLLM proxy (recommended) or local model

Spanish cue text is translated by the first engine that is configured and healthy:

1. **LiteLLM / OpenAI-compatible endpoint** (`TRANSLATION_ENDPOINT` + `TRANSLATION_API_KEY`), using the resolved source language.
2. **Bundled English → Spanish local model** (automatic fallback for English, or the only engine when no endpoint/key is set).

Non-English sources never fall back to the English-only model: if the proxy is unavailable or fails, the API returns `503` with no subtitles.

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

**Reliability.** Each batch is a numbered JSON request sent with a strict JSON-schema `response_format`, and a reply is accepted only when it has exactly one non-empty translation per requested cue index; otherwise it is retried once and, for English sources with fallback enabled, handed to the local model. Reordered replies are re-ordered by index rather than discarded. Malformed shapes (missing keys, boolean/fractional/duplicate indices, blank or truncated content) are rejected instead of crashing or misaligning cues. Batches are 16 cues by default (`TRANSLATION_BATCH_CUES`, max 64) with a bounded `TRANSLATION_MAX_TOKENS`, so a long episode cannot ask for one oversized completion. Each batch logs its file, batch number, engine and retry reasons — never subtitle text, credentials or upstream response bodies. Failures never yield English subtitles labelled as Spanish.

**Source-language resolution.** Spanish translation requests use a priority-based evidence chain: a recognized language token in the supplied filename (for example `German.Dubbed`), then Whisper acoustic detection, then a lightweight transcript word classifier, finally English as a fallback. This is not a majority vote: filename hints take precedence over acoustic detection, and the lexical check does not override either usable hint. The API's `language` argument does not force the source for Spanish translation. Logs record `source_lang`, `acoustic` and the deciding `evidence`; the supplied media path is never opened.

**Language settings (Unix/POSIX).** Locale settings are validated at start-up. For neutral Latin American Spanish, set these in `.env` or export them before starting Compose:

```bash
TRANSLATION_SOURCE_LOCALE=en_US.UTF-8     # English local-model source setting
TRANSLATION_TARGET_LOCALE=es_419.UTF-8    # neutral Latin American Spanish
```

`es_419.UTF-8` and `es-419` are accepted, as are `es_MX.UTF-8`, `es-MX`, `es_MX` and bare `es`. **The current code and Compose default remains `es_MX.UTF-8`; set `es_419.UTF-8` explicitly to use the Latin America locale label.** The proxy prompt favors neutral Latin American vocabulary and grammar (`computadora`, `celular`, `ustedes`, not `ordenador`, `móvil`, `vosotros`). `/status` and `/health` report the effective locales; `source_locale` is a configuration value, not the per-request detected source.

**The offline fallback cannot regionalise**: OPUS-MT is trained on generic Spanish, so its batches log `engine=local locale_fidelity=generic_Spanish`. An explicit regional target must match the configured target locale (for example, `target_language=es_ES.UTF-8` is rejected with `400` when configured for `es_419.UTF-8`). Bare `target_language=es` uses the configured target. The text locale does not change Bazarr's profile or subtitle tag; see [conflicting language rules](#search-completes-without-downloading-conflicting-language-rules).

## Matching and scores

Bazarr scores candidate subtitles by comparing release metadata. A generated subtitle is produced **from the exact input file**, so it is bound to that path and can legitimately claim matching source fields:

- The result is tied to the input path, and only **year and release group** are snapshotted from that same file for additional matching points; source, resolution, codecs, streaming service and edition are deliberately excluded.
- Only fields that are **populated and equal** are claimed; a result generated from one episode is never scored against another path.
- The release column shows the file name so it is obvious which episode the generated subtitle came from.
- **No hash match, hearing-impaired, forced or quality claim is fabricated.** `hash_verifiable` stays `false`.

For episodes, the built-in series/season/episode matches score **220/360 (~61%)**. Matching year adds 90 and release group adds 20, yielding at most **330/360 (~92%)**. This deliberately limited match set clears a 90% minimum when both fields match while staying below a real release scoring around 93%, so that release can be preferred. Missing metadata lowers the score; this is not a guaranteed 92% score or a movie-score ceiling.

> **Caveat:** this is a *release-matching* score, not a translation-quality measurement. A high score means Bazarr is confident the subtitle belongs to that episode — it says nothing about how good the Spanish is. Keep manual review; do not lower `minimum_score` globally just because one provider scores high.

## API smoke test

From a container on Bazarr's network, given a short PCM test clip:

```bash
curl -f -F audio_file=@clip.pcm \
  'http://whisper-arc:9000/asr?encode=false&language=en&task=translate&target_language=es&output=srt'
curl -f -F audio_file=@clip.pcm \
  'http://whisper-arc:9000/detect-language?encode=false'
```

Omitting `target_language` preserves native Whisper behavior. Spanish targets require `task=translate`; `language` may be omitted or supplied, but the adapter resolves the source from filename/acoustic/transcript evidence. Non-English translation requires the proxy. Other targets or incompatible task/target combinations return 400. Translation errors return 503 with no subtitles, never untranslated text mislabeled as Spanish. Concurrent requests wait for the single pipeline and then run in order (see [Diagnosing failed downloads](#diagnosing-failed-downloads)); they only fail with 503 + `Retry-After` after the queue wait expires or the bounded queue is full.

## Diagnosing failed downloads

Every request logs one greppable line naming the media file Bazarr passed as `video_file`, plus the task, language, duration and result. `path` is only echoed from Bazarr; the service still never opens a media path.

```bash
docker logs whisper-arc 2>&1 | tail -50              # recent requests
docker logs whisper-arc 2>&1 | grep 'Show.S01E03'    # one episode by file name
docker logs whisper-arc 2>&1 | grep 'asr failed'     # every failed request
docker logs whisper-arc 2>&1 | grep 'asr empty'      # transcriptions with no cues
docker logs whisper-arc 2>&1 | grep 'queued'         # requests that had to wait
```

Sample lines (one per state):

```text
asr start file="Show.S01E03.mkv" path="/data/media/series/99/Season 01/Show.S01E03.mkv" task=translate language=en target=es encode=false
asr queued file="Show.S01E03.mkv" path="..." waited_seconds=41.2
language resolved file="Show.S01E03.mkv" source_lang=en acoustic=en evidence=whisper_acoustic
asr transcribed file="Show.S01E03.mkv" cues=812 source_lang=en
translation ok file="Show.S01E03.mkv" source_lang=en evidence=whisper_acoustic
asr ok file="Show.S01E03.mkv" path="..." cues=812 bytes_out=41234 seconds=97.4
asr empty file="Show.S01E05.mkv" path="..." cues=0 bytes_out=0 seconds=63.1
asr failed file="Show.S01E04.mkv" path="..." status=503 reason=Whisper queue wait timed out seconds=900.0
detect ok file="Show.S01E06.mkv" path="..." language=en seconds=1.2
```

How to read the result:

- `asr failed`/`detect failed` carry the exact HTTP status and reason: `503` can mean a queue/timeout problem or translation failure; `400`/`413` indicates invalid parameters, malformed audio or oversized input. GPU/model initialization errors may return `500`; `/health` returns `503` when its GPU/model availability check fails. Translation logs distinguish proxy retries, local fallback and unsupported non-English fallback.
- `asr empty` means the audio produced no cues (silence, music or an all-speech-off episode); Bazarr receives a successful but empty subtitle.
- **No line at all for an episode** means Bazarr never called this service. Check the Bazarr log for the same file name: the provider can refuse before upload for unsupported target languages, a Spanish translation on an endpoint other than the configured Arc endpoint (the hook includes the full path in that message), or an audio track with no usable language when detection fails.

### Search completes without downloading: conflicting language rules

For a **Spanish** (`es`) profile, use only **Spanish (Latino) → Spanish** in
Bazarr's Settings → Languages → Language Equals (`spl:spa`). Do not add the
reverse `spa:spl` rule: these rules rewrite returned subtitle languages. With
both directions, ordinary Spanish results from Whisper and direct providers
become `es-MX` and are rejected as `Skipping, language not searched for` before
scoring or download. Removing the reverse rule was verified with an actual
episode download from OpenSubtitles, synchronization, and a cleared missing flag.

Bazarr displays `es-MX` as **Spanish (Latino)**. When configured, the adapter's
`es_419.UTF-8` locale controls translation text, not the Bazarr profile/tag; changing the text
locale cannot fix this selection failure. Language-equals rules apply to provider
results and do not automatically relabel existing files. Search APIs enqueue
work: wait for the job, then check history and the indexed subtitle file rather
than treating HTTP 204 or a completed search card as proof of a download.
Disable diagnostic debug logging afterward: provider HTTP logs can contain API
credentials.

Tuning (all optional, defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_LOG_LEVEL` | `INFO` | `DEBUG` also logs Bazarr-facing details; `WARNING` silences request lines |
| `WHISPER_GATE_WAIT_SECONDS` | `900` | How long a queued request waits for the busy pipeline before returning 503 |
| `WHISPER_MAX_WAITERS` | `4` | Requests allowed to queue; beyond it the answer is 503 + `Retry-After: 30` |

Keep `WHISPER_GATE_WAIT_SECONDS` below Bazarr's provider read timeout, otherwise Bazarr gives up before the queue does.

## Frequently asked questions

### Can Bazarr generate subtitles locally with an Intel Arc GPU?

Yes. Configure Bazarr's Whisper provider to use this service. OpenVINO runs Whisper speech recognition on Intel Arc; the service returns SRT subtitles for Bazarr to save. An Arc A380 has been tested; other GPUs are not validated here.

### Can I translate subtitles into Spanish without a cloud API?

Yes, for English audio: Whisper transcription and the bundled English → Spanish model run locally. Other source languages require the optional translation proxy, which receives cue text but not audio, media filenames or timestamps.

### Do I need an NVIDIA GPU or CUDA?

No. This project uses Intel Arc and OpenVINO, not CUDA. Docker needs access to `/dev/dri` and working Intel GPU drivers on the host; Whisper inference does not fall back to CPU.

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
- Spanish is the only text-translation target. Non-English sources require the proxy; the local model handles English → generic Spanish only. The proxy prompt favors neutral Latin American Spanish; locale selection is not a guarantee of regional fidelity.
- Automated Bazarr score is release matching, not translation quality.
- One inference request at a time, 2 CPU / 2 GiB cap in Compose. Extra requests queue in order (bounded by `WHISPER_GATE_WAIT_SECONDS`/`WHISPER_MAX_WAITERS`) instead of failing immediately, so parallel Bazarr searches serialize. First GPU initialization is slower. The adapter retains its models in memory.
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
