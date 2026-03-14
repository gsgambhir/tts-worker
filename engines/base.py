"""Base interface for TTS engines."""

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """All TTS engines implement this interface."""

    @abstractmethod
    def generate(
        self,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> tuple[bytes, int]:
        """
        Generate speech audio from text.

        Returns (wav_bytes, sample_rate).
        """

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """Return available voices as a list of dicts with 'id' and 'name'."""
