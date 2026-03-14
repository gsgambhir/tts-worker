"""Dia TTS engine — 1.6B params, multi-speaker dialogue, non-verbal sounds."""

import io

import numpy as np
import soundfile as sf

from engines.base import TTSEngine


class DiaEngine(TTSEngine):
    def __init__(self):
        from dia.model import Dia

        print("Loading Dia engine...")
        self.model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B-0626",
            compute_dtype="float16",
        )
        print("Dia engine ready.")

    def generate(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[bytes, int]:
        # voice param is ignored for Dia — use [S1]/[S2] tags in text
        audio_prompt = kwargs.get("audio_prompt")

        output = self.model.generate(
            text,
            use_torch_compile=False,
            cfg_scale=kwargs.get("cfg_scale", 3.0),
            temperature=kwargs.get("temperature", 1.3),
            top_p=kwargs.get("top_p", 0.95),
            audio_prompt=audio_prompt,
            verbose=False,
        )

        buf = io.BytesIO()
        sf.write(buf, output, 44100, format="WAV")
        return buf.getvalue(), 44100

    def list_voices(self) -> list[dict]:
        return [
            {"id": "dialogue", "name": "Multi-speaker dialogue (use [S1]/[S2] tags)"},
        ]
