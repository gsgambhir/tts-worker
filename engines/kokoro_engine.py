"""Kokoro TTS engine — 82M params, ultra-fast, 54 voices, 9 languages."""

import io

import numpy as np
import soundfile as sf

from engines.base import TTSEngine

KOKORO_VOICES = {
    # American English
    "af_heart": "American Female - Heart",
    "af_alloy": "American Female - Alloy",
    "af_aoede": "American Female - Aoede",
    "af_bella": "American Female - Bella",
    "af_jessica": "American Female - Jessica",
    "af_kore": "American Female - Kore",
    "af_nicole": "American Female - Nicole",
    "af_nova": "American Female - Nova",
    "af_river": "American Female - River",
    "af_sarah": "American Female - Sarah",
    "af_sky": "American Female - Sky",
    "am_adam": "American Male - Adam",
    "am_echo": "American Male - Echo",
    "am_eric": "American Male - Eric",
    "am_fenrir": "American Male - Fenrir",
    "am_liam": "American Male - Liam",
    "am_michael": "American Male - Michael",
    "am_onyx": "American Male - Onyx",
    "am_puck": "American Male - Puck",
    # British English
    "bf_alice": "British Female - Alice",
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bf_lily": "British Female - Lily",
    "bm_daniel": "British Male - Daniel",
    "bm_fable": "British Male - Fable",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
    # Japanese
    "jf_alpha": "Japanese Female - Alpha",
    "jm_kumo": "Japanese Male - Kumo",
    # Chinese
    "zf_xiaobei": "Chinese Female - Xiaobei",
    "zf_xiaoni": "Chinese Female - Xiaoni",
    "zm_yunjian": "Chinese Male - Yunjian",
    "zm_yunxi": "Chinese Male - Yunxi",
    # Spanish
    "ef_dora": "Spanish Female - Dora",
    "em_alex": "Spanish Male - Alex",
    # French
    "ff_siwis": "French Female - Siwis",
    # Hindi
    "hf_alpha": "Hindi Female - Alpha",
    "hm_omega": "Hindi Male - Omega",
    # Italian
    "if_sara": "Italian Female - Sara",
    "im_nicola": "Italian Male - Nicola",
    # Brazilian Portuguese
    "pf_dora": "Portuguese Female - Dora",
    "pm_alex": "Portuguese Male - Alex",
}

# Map language prefix to lang_code
LANG_MAP = {
    "a": "a", "b": "b", "j": "j", "z": "z",
    "e": "e", "f": "f", "h": "h", "i": "i", "p": "p",
}


class KokoroEngine(TTSEngine):
    def __init__(self):
        from kokoro import KPipeline

        self._pipelines: dict = {}
        self._KPipeline = KPipeline

        # Pre-load American English pipeline
        print("Loading Kokoro engine (American English)...")
        self._pipelines["a"] = KPipeline(lang_code="a", device="cuda")
        print("Kokoro engine ready.")

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            print(f"Loading Kokoro pipeline for language: {lang_code}")
            self._pipelines[lang_code] = self._KPipeline(
                lang_code=lang_code, device="cuda"
            )
        return self._pipelines[lang_code]

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[bytes, int]:
        # Determine language from voice prefix
        lang_code = LANG_MAP.get(voice[0], "a") if voice else "a"
        pipeline = self._get_pipeline(lang_code)

        chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio)

        if not chunks:
            return b"", 24000

        full_audio = np.concatenate(chunks)

        buf = io.BytesIO()
        sf.write(buf, full_audio, 24000, format="WAV")
        return buf.getvalue(), 24000

    def list_voices(self) -> list[dict]:
        return [
            {"id": vid, "name": name}
            for vid, name in KOKORO_VOICES.items()
        ]
