# syntax=docker/dockerfile:1

FROM ubuntu:24.04

ARG BUILD_DATE
ARG VERSION
LABEL build_version="${VERSION}" build_date="${BUILD_DATE}"

ENV DEBIAN_FRONTEND="noninteractive" \
    TMPDIR="/run/whisper-temp"

ARG WHISPER_MODEL=small

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
    /opt/venv/bin/python3 -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8', download_root='/config')" && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/*

COPY whisper_typer/server/ /app/

WORKDIR /app

VOLUME /config

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "transcribe_api:app", "--host", "0.0.0.0", "--port", "8000"]
