# 🔊 TTSS - Text-to-Speech for ComfyUI

Multi-engine Text-to-Speech nodes for ComfyUI with voice cloning support.

## Features

- **🎤 Multiple TTS Engines**:
  - `pyttsx3` - Offline, uses system voices (Windows SAPI, macOS, Linux espeak)
  - `edge-tts` - Microsoft Edge TTS (online, high quality, free, 400+ voices)
  - `xtts-v2` - Neural TTS with voice cloning via Auralis (GPU, Python 3.10+)

- **🎙️ Voice Cloning**: Clone any voice using reference audio (XTTS-v2)
- **🌍 Multi-language**: 100+ languages supported via edge-tts
- **📝 SRT Support**: Read subtitles directly for TTS
- **🔗 Audio Combining**: Merge multiple audio files with crossfade

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "TTSS" or paste: `https://github.com/Kraven1109/TTSS.git`
3. Click Install and restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Kraven1109/TTSS.git comfyUI-TTSS
cd comfyUI-TTSS
pip install -r requirements.txt
```

## Nodes

| Node | Icon | Description |
|------|------|-------------|
| `TTSSTextToSpeech` | 🔊 | Main TTS node - supports all 3 engines with built-in voice selection |
| `TTSSLoadReferenceAudio` | 🎙️ | Load reference audio for voice cloning |
| `TTSSLoadAudio` | 📂 | Load audio from input directory |
| `TTSSLoadSRT` | 📄 | Load SRT subtitle file |
| `TTSSPreviewAudio` | 🎧 | Preview audio in ComfyUI UI |
| `TTSSCombineAudio` | 🔗 | Combine multiple audio files |
| `TTSSSaveAudio` | 💾 | Save audio with custom filename/format |

## TTS Engines

### 1. pyttsx3 (Offline)
- ✅ Works offline, no internet required
- ✅ Uses system voices (Windows SAPI, macOS NSSpeechSynthesizer)
- ⚠️ Limited voice quality compared to neural TTS

### 2. edge-tts (Recommended)
- ✅ High quality Microsoft neural voices
- ✅ 550+ voices in 100+ languages
- ✅ Free, no API key required
- ⚠️ Requires internet connection

**Popular voices:**
- `en-US-AriaNeural` - Female, natural
- `en-US-GuyNeural` - Male, natural
- `vi-VN-HoaiMyNeural` - Vietnamese female
- `ja-JP-NanamiNeural` - Japanese female
- `zh-CN-XiaoxiaoNeural` - Chinese female

### 3. xtts-v2 (Neural + Voice Cloning via Auralis)
- ✅ Highest quality neural TTS with voice cloning
- ✅ Works with Python 3.10+ (ComfyUI compatible!)
- ✅ Clone any voice with just 6 seconds of audio
- ✅ 17 languages supported
- ⚠️ Requires GPU and `pip install auralis`
- ⚠️ Requires reference audio for synthesis

**Why Auralis instead of Coqui TTS?**
- Coqui TTS requires Python <3.12, but ComfyUI uses Python 3.12/3.13
- Auralis wraps XTTS-v2 and works with Python 3.10+
- Same XTTS-v2 quality, modern Python support!

## Directory Structure

```
ComfyUI/
├── models/
│   └── tts/                    # TTS model directory
│       ├── reference_audio/    # Voice cloning reference files (.wav, 6+ seconds)
│       ├── xtts/               # XTTS-v2 / Auralis models
│       └── voices/             # Custom voice models
├── input/                      # Audio/SRT files for loading
└── output/                     # Generated audio output
```

## Usage Examples

### Basic Text-to-Speech
```
[🔊 Text to Speech] → [🎧 Preview Audio]
     ↳ text: "Hello world"
     ↳ engine: edge-tts
     ↳ edge_voice: en-US-AriaNeural
```

### Voice Cloning (XTTS-v2)
```
[🎙️ Load Reference Audio] ── reference_audio ──→ [🔊 Text to Speech] → [🎧 Preview Audio]
                                                      ↳ engine: xtts-v2
```

### With ComfyUI-LLama (Image to Speech)
```
[LoadImage] → [🦙 LLama Server] ── text ──→ [🔊 Text to Speech] → [🎧 Preview Audio]
                                                ↳ text_input
```

### SRT Subtitle to Audio
```
[📄 Load SRT] ── srt_path ──→ [🔊 Text to Speech] → [💾 Save Audio]
                                   ↳ srt_input
```

### Combine Multiple Audio
```
[🔊 TTS 1] ──→ audio1 ──┐
[🔊 TTS 2] ──→ audio2 ──┼──→ [🔗 Combine Audio] → [🎧 Preview Audio]
[🔊 TTS 3] ──→ audio3 ──┘
```

## Node I/O Reference

### TTSSTextToSpeech 🔊
**Inputs:**
- `text` (STRING) - Text to synthesize
- `engine` (dropdown) - pyttsx3 / edge-tts / xtts-v2
- `speed` (FLOAT) - 0.5 to 2.0 (pyttsx3 & edge-tts only)
- `edge_voice` (dropdown) - Select from 550+ Microsoft Edge voices
- `pyttsx3_voice` (dropdown) - Select system voice
- `xtts_model` (dropdown) - Select XTTS-v2 model
- `text_input` (STRING, optional) - Piped text input
- `srt_input` (SRT, optional) - SRT file path
- `reference_audio` (AUDIOPATH, optional) - For XTTS-v2 voice cloning

**Outputs:**
- `audio_path` (AUDIOPATH)

## Requirements

### Minimal (pyttsx3 only)
```
pyttsx3>=2.90
pydub>=0.25.1
```

### Recommended (with edge-tts)
```
pyttsx3>=2.90
pydub>=0.25.1
edge-tts>=6.1.0
```

### Full (with XTTS-v2 voice cloning via Auralis)
```
pyttsx3>=2.90
pydub>=0.25.1
edge-tts>=6.1.0
auralis>=0.2.8    # XTTS-v2 for Python 3.10+ (works with ComfyUI!)
torch>=2.0.0
```

**Note:** pydub requires ffmpeg. Install via:
- Windows: `conda install ffmpeg` or download from ffmpeg.org
- Linux: `apt install ffmpeg`
- macOS: `brew install ffmpeg`

## Related Projects

- **[ComfyUI-LLama](https://github.com/Kraven1109/ComfyUI-Llama)** - LLM inference for image description → TTS pipeline

## License

MIT License

## Acknowledgements

- [Auralis](https://github.com/astramind-ai/Auralis) - XTTS-v2 wrapper for Python 3.10+
- [edge-tts](https://github.com/rany2/edge-tts) - Microsoft Edge TTS wrapper
- [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) - Voice cloning neural TTS
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - Offline TTS
