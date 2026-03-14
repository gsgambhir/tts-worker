# TTS Multi-Engine RunPod Worker

RunPod Serverless worker for text-to-speech with three engines:

- **Kokoro** — 82M params, 54 voices, 9 languages, 36-96x realtime. Apache 2.0.
- **Dia** — 1.6B params, multi-speaker dialogue with non-verbal sounds. Apache 2.0.
- **F5-TTS** — 300M params, zero-shot voice cloning from 5-15s audio sample.

## Quick Start

Set `TTS_ENGINE` to `kokoro`, `dia`, or `f5` when creating your endpoint.

```bash
docker build --platform linux/amd64 -t yourusername/tts-worker:latest .
docker push yourusername/tts-worker:latest
```

## Input Formats

### Kokoro (OpenAI-compatible)
```json
{
  "input": {
    "input": "Hello, world!",
    "voice": "af_heart",
    "speed": 1.0,
    "response_format": "wav"
  }
}
```

### Dia (Multi-speaker dialogue)
```json
{
  "input": {
    "input": "[S1] Hey, how are you? [S2] I'm great! (laughs) Thanks for asking.",
    "model": "dia",
    "temperature": 1.3
  }
}
```

### F5-TTS (Voice cloning)
```json
{
  "input": {
    "input": "Text to speak in cloned voice.",
    "ref_audio": "<base64-encoded WAV of reference voice>",
    "ref_text": "Transcript of the reference audio."
  }
}
```

### List Voices
```json
{
  "input": {
    "action": "list_voices"
  }
}
```

## Output

All engines return:
```json
{
  "audio": "<base64-encoded audio>",
  "format": "wav",
  "sample_rate": 24000,
  "engine": "kokoro"
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `kokoro` | Engine to use: `kokoro`, `dia`, or `f5` |

## Voices (Kokoro)

54 voices across American English, British English, Japanese, Chinese, Spanish, French, Hindi, Italian, and Brazilian Portuguese. Use `{"action": "list_voices"}` to get the full list.

## License

- Kokoro: Apache 2.0
- Dia: Apache 2.0
- F5-TTS: MIT (code), CC-BY-NC-4.0 (pretrained weights — non-commercial)
