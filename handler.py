"""
RunPod Serverless TTS Handler.

Supports three engines (Kokoro, Dia, F5-TTS) with an OpenAI-compatible
/v1/audio/speech input format. Select engine via TTS_ENGINE env var.
"""

import base64
import io
import os
import tempfile

import runpod
import soundfile as sf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TTS_ENGINE = os.environ.get("TTS_ENGINE", "kokoro")

# ---------------------------------------------------------------------------
# Load the selected engine at startup
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"TTS RunPod Worker — Loading engine: {TTS_ENGINE}")
print("=" * 60)

if TTS_ENGINE == "kokoro":
    from engines.kokoro_engine import KokoroEngine
    engine = KokoroEngine()
elif TTS_ENGINE == "dia":
    from engines.dia_engine import DiaEngine
    engine = DiaEngine()
elif TTS_ENGINE == "f5":
    from engines.f5_engine import F5Engine
    engine = F5Engine()
else:
    raise ValueError(f"Unknown TTS_ENGINE: {TTS_ENGINE}. Use 'kokoro', 'dia', or 'f5'.")


def _convert_audio(wav_bytes: bytes, sr: int, fmt: str) -> bytes:
    """Convert WAV bytes to the requested output format."""
    if fmt == "wav":
        return wav_bytes

    audio_data, _ = sf.read(io.BytesIO(wav_bytes))

    buf = io.BytesIO()
    if fmt == "flac":
        sf.write(buf, audio_data, sr, format="FLAC")
    elif fmt == "pcm":
        import numpy as np
        pcm = (audio_data * 32767).astype(np.int16)
        buf.write(pcm.tobytes())
    else:
        # For mp3/opus, fall back to WAV (mp3 encoding needs pydub/ffmpeg)
        sf.write(buf, audio_data, sr, format="WAV")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """
    OpenAI-compatible TTS input format:
    {
        "input": "Text to speak",
        "voice": "af_heart",
        "model": "kokoro",
        "speed": 1.0,
        "response_format": "wav"
    }

    Dia-specific fields:
    {
        "input": "[S1] Hello! [S2] Hi there! (laughs)",
        "model": "dia",
        "cfg_scale": 3.0,
        "temperature": 1.3,
        "audio_prompt": "<base64 WAV for voice conditioning>"
    }

    F5-TTS-specific fields:
    {
        "input": "Text to speak in cloned voice",
        "model": "f5",
        "ref_audio": "<base64 WAV of reference voice>",
        "ref_text": "Transcript of the reference audio"
    }
    """
    job_input = job["input"]

    text = job_input.get("input", "")
    if not text:
        return {"error": "Missing 'input' field (text to synthesize)"}

    voice = job_input.get("voice", "af_heart")
    speed = float(job_input.get("speed", 1.0))
    response_format = job_input.get("response_format", "wav")

    # Build engine-specific kwargs
    kwargs = {}

    # Dia: voice conditioning via base64 audio
    if TTS_ENGINE == "dia":
        if "audio_prompt" in job_input:
            # Save base64 audio to temp file
            audio_bytes = base64.b64decode(job_input["audio_prompt"])
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            kwargs["audio_prompt"] = tmp.name

        for key in ("cfg_scale", "temperature", "top_p"):
            if key in job_input:
                kwargs[key] = float(job_input[key])

    # F5-TTS: reference audio for voice cloning
    if TTS_ENGINE == "f5":
        if "ref_audio" in job_input:
            audio_bytes = base64.b64decode(job_input["ref_audio"])
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            kwargs["ref_audio_path"] = tmp.name
        else:
            return {"error": "F5-TTS requires 'ref_audio' (base64 WAV) for voice cloning"}

        kwargs["ref_text"] = job_input.get("ref_text", "")
        if "nfe_step" in job_input:
            kwargs["nfe_step"] = int(job_input["nfe_step"])
        if "seed" in job_input:
            kwargs["seed"] = int(job_input["seed"])

    try:
        wav_bytes, sample_rate = engine.generate(
            text=text,
            voice=voice,
            speed=speed,
            **kwargs,
        )
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Clean up temp files
        for key in ("audio_prompt", "ref_audio_path"):
            path = kwargs.get(key)
            if path and os.path.exists(path):
                os.unlink(path)

    if not wav_bytes:
        return {"error": "No audio generated"}

    output_bytes = _convert_audio(wav_bytes, sample_rate, response_format)

    return {
        "audio": base64.b64encode(output_bytes).decode("utf-8"),
        "format": response_format,
        "sample_rate": sample_rate,
        "engine": TTS_ENGINE,
    }


# ---------------------------------------------------------------------------
# Voices endpoint (via openai_route passthrough)
# ---------------------------------------------------------------------------
def dispatch(job):
    job_input = job["input"]

    # Support a "list_voices" action
    if job_input.get("action") == "list_voices":
        return {"voices": engine.list_voices(), "engine": TTS_ENGINE}

    return handler(job)


runpod.serverless.start({"handler": dispatch})
