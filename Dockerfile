# =============================================================================
# Piper TTS — CPU-only RunPod Serverless worker
# Small (~300 MB), no GPU. Uses the official Linux x86_64 Piper binary
# (the Linux build ships its shared libs intact, unlike the macOS build).
# =============================================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Piper Linux binary (bundles libonnxruntime, libespeak-ng, libpiper_phonemize)
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    && tar xzf piper_linux_x86_64.tar.gz \
    && rm piper_linux_x86_64.tar.gz

# en-US voice baked into the image (no runtime download -> fast cold start)
RUN mkdir -p voices \
    && wget -q -O voices/en_US-lessac-medium.onnx \
       "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx?download=true" \
    && wget -q -O voices/en_US-lessac-medium.onnx.json \
       "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json?download=true"

RUN pip install --no-cache-dir runpod

COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV PIPER_BIN=/app/piper/piper
ENV PIPER_VOICE=/app/voices/en_US-lessac-medium.onnx

CMD ["python", "-u", "handler.py"]
