# syntax=docker/dockerfile:1

FROM ubuntu:24.04

ARG BUILD_DATE
ARG VERSION
LABEL build_version="${VERSION}" build_date="${BUILD_DATE}"

ENV DEBIAN_FRONTEND="noninteractive" \
    TMPDIR="/run/whisper-temp"

ARG WHISPER_MODEL=tiny

RUN mkdir -p /run/whisper-temp && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        curl && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip wheel && \
    /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn python-multipart faster-whisper numpy && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/*

COPY root/app/ /app/

WORKDIR /app

# Download model at build time so it's baked into the image layer.
# Pass --build-arg WHISPER_MODEL=small (etc.) to override.
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    WHISPER_MODEL="${WHISPER_MODEL}" \
    /opt/venv/bin/python -c "\
import os; from faster_whisper import WhisperModel; \
m = os.environ.get('WHISPER_MODEL', 'tiny'); \
print(f\"Downloading Whisper model '{m}'...\"); \
WhisperModel(m, device='cpu', compute_type='int8', download_root='/config'); \
print('Done.')"

VOLUME /config

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "transcribe_api:app", "--host", "0.0.0.0", "--port", "8000"]
