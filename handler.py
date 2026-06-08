"""
RunPod Serverless Piper TTS handler (CPU-only).

Input  : { "input": { "input": "<text>", "length_scale": 1.0 } }
         ("text" is also accepted as an alias for "input")
Output : { "audio": "<base64 WAV>", "format": "wav", "sample_rate": 22050 }

length_scale controls pace: 1.0 = normal, >1.0 = slower, <1.0 = faster.
"""

import base64
import os
import subprocess
import tempfile

import runpod

PIPER_BIN = os.environ.get("PIPER_BIN", "/app/piper/piper")
PIPER_VOICE = os.environ.get("PIPER_VOICE", "/app/voices/en_US-lessac-medium.onnx")
SAMPLE_RATE = 22050  # en_US-lessac-medium is 22.05 kHz


def handler(job):
    job_input = job.get("input", {}) or {}

    text = job_input.get("input") or job_input.get("text") or ""
    if not text:
        return {"error": "Missing 'input' field (text to synthesize)"}

    try:
        length_scale = float(job_input.get("length_scale", 1.0))
    except (TypeError, ValueError):
        length_scale = 1.0

    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        proc = subprocess.run(
            [
                PIPER_BIN,
                "--model", PIPER_VOICE,
                "--length_scale", str(length_scale),
                "--output_file", out_path,
            ],
            input=text.encode("utf-8"),
            capture_output=True,
        )
        if proc.returncode != 0:
            return {"error": proc.stderr.decode("utf-8", "ignore")[:800]}

        with open(out_path, "rb") as fh:
            audio = fh.read()

        if not audio:
            return {"error": "Piper produced no audio"}

        return {
            "audio": base64.b64encode(audio).decode("utf-8"),
            "format": "wav",
            "sample_rate": SAMPLE_RATE,
        }
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


runpod.serverless.start({"handler": handler})
