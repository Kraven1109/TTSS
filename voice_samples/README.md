# Voice Samples for TTSS

Place your voice reference audio files here for voice cloning presets.

## File naming convention

The files should match the preset names in `nodes.py`:
- `male_en.wav` - Male English voice
- `female_en.wav` - Female English voice
- `male_zh.wav` - Male Chinese voice
- `female_zh.wav` - Female Chinese voice

## Requirements

- **Format**: WAV (recommended), MP3, FLAC, M4A
- **Duration**: 6-15 seconds of clear speech
- **Quality**: Clean audio without background noise
- **Content**: Natural speech in the target language

## How XTTS voice cloning works

XTTS-v2 uses audio reference samples to clone the voice characteristics:
- Pitch and tone
- Speaking style
- Accent and pronunciation patterns

The model does NOT use built-in male/female voices. Instead, it learns from your reference audio.

## Tips for best results

1. Use high-quality microphone recordings
2. Avoid reverb, echo, or background noise
3. Include varied intonation (questions, statements)
4. Keep audio between 6-15 seconds
5. Use the same language as your target output

## Example sources

You can find free voice samples at:
- LibriSpeech dataset
- Common Voice by Mozilla
- VCTK corpus
