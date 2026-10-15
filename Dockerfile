# Tested Intel GPU runtime; run our adapter, not the base image's application.
FROM ventura8/whisper-pro-asr@sha256:cffd42a6ed58fa9a05d2abe11fb43a96cd2a31499d8542aab5b964d8ea3d3fbf
USER root
RUN /opt/venv/bin/pip install --no-cache-dir sentencepiece==0.2.1
COPY src/ /srv/whisper-arc/
COPY LICENSE /srv/whisper-arc/LICENSE
USER 65534:65534
ENV HOME=/tmp WHISPER_MODEL_DIR=/models TRANSLATION_MODEL_DIR=/translation-model
WORKDIR /srv/whisper-arc
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
