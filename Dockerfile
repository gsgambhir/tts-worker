# =============================================================================
# TTS Worker — Kokoro, Dia, F5-TTS on RunPod Serverless
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    espeak-ng ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.6 first (pinned for Dia compatibility)
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.6.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Install TTS engines
RUN pip install --no-cache-dir --break-system-packages \
    kokoro>=0.9.4 \
    f5-tts \
    && pip install --no-cache-dir --break-system-packages \
    git+https://github.com/nari-labs/dia.git

# Install RunPod SDK and utilities
RUN pip install --no-cache-dir --break-system-packages \
    runpod>=1.7.0 soundfile numpy requests

# Copy application code
COPY engines/ engines/
COPY handler.py .

ENV PYTHONUNBUFFERED=1

# Default: Kokoro (lightest, fastest). Override with TTS_ENGINE=dia or TTS_ENGINE=f5
ENV TTS_ENGINE="kokoro"

CMD ["python3", "-u", "handler.py"]
