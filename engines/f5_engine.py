"""F5-TTS engine — ~300M params, zero-shot voice cloning."""

import io
import os
import tempfile

import numpy as np
import soundfile as sf

from engines.base import TTSEngine


class F5Engine(TTSEngine):
    def __init__(self):
        from f5_tts.api import F5TTS

        print("Loading F5-TTS engine...")
        self.model = F5TTS(
            model="F5TTS_v1_Base",
            device="cuda",
        )
        print("F5-TTS engine ready.")

    def generate(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[bytes, int]:
        ref_audio_path = kwargs.get("ref_audio_path")
        ref_text = kwargs.get("ref_text", "")

        if not ref_audio_path:
            raise ValueError(
                "F5-TTS requires a reference audio file. "
                "Provide 'ref_audio' (base64 WAV) in the request."
            )

        # Auto-transcribe if no ref_text
        if not ref_text:
            ref_text = self.model.transcribe(ref_audio_path)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            wav, sr, _ = self.model.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=text,
                file_wave=output_path,
                speed=speed,
                nfe_step=kwargs.get("nfe_step", 32),
                seed=kwargs.get("seed"),
            )

            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV")
            return buf.getvalue(), sr
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def list_voices(self) -> list[dict]:
        return [
            {"id": "clone", "name": "Voice cloning (provide ref_audio)"},
        ]
