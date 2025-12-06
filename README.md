# 🔊 TTSS - Text-to-Speech for ComfyUI

Multi-engine Text-to-Speech nodes for ComfyUI with voice cloning support.

## Features

- **🎤 Multiple TTS Engines**:
  - `pyttsx3` - Offline, uses system voices (Windows SAPI, macOS, Linux espeak)
  - `edge-tts` - Microsoft Edge TTS (online, high quality, free, 400+ voices)
  - `coqui-tts` - Neural TTS with voice cloning (local GPU)

- **🎙️ Voice Cloning**: Clone any voice using reference audio
- **🌍 Multi-language**: 100+ languages supported via edge-tts
- **📝 SRT Support**: Read subtitles directly for TTS
- **🔗 Audio Combining**: Merge multiple audio files with crossfade

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "TTSS" or paste: `https://github.com/your-username/comfyUI-TTSS.git`
3. Click Install and restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/comfyUI-TTSS.git
cd comfyUI-TTSS
pip install -r requirements.txt
```

## Nodes

| Node | Icon | Description |
|------|------|-------------|
| `TTSSTextToSpeech` | 🔊 | Main TTS node - supports all 3 engines |
| `TTSSVoiceSelector` | 🎤 | Select voice from dropdown by engine |
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
- ✅ 400+ voices in 100+ languages
- ✅ Free, no API key required
- ⚠️ Requires internet connection

**Popular voices:**
- `en-US-AriaNeural` - Female, natural
- `en-US-GuyNeural` - Male, natural
- `vi-VN-HoaiMyNeural` - Vietnamese female
- `ja-JP-NanamiNeural` - Japanese female
- `zh-CN-XiaoxiaoNeural` - Chinese female

### 3. coqui-tts (Neural + Voice Cloning)
- ✅ Highest quality neural TTS
- ✅ Voice cloning with reference audio
- ⚠️ Requires GPU and `pip install TTS`
- ⚠️ Larger model download

## Directory Structure

```
ComfyUI/
├── models/
│   └── tts/                    # TTS model directory
│       ├── reference_audio/    # Voice cloning reference files
│       ├── coqui/              # Coqui TTS models
│       └── voices/             # Custom voice models
├── input/                      # Audio/SRT files for loading
└── output/                     # Generated audio output
```

## Usage Examples

### Basic Text-to-Speech
```
[📝 Text Input] → [🔊 Text to Speech] → [🎧 Preview Audio]
```

### With Voice Selection
```
[🎤 Voice Selector] → [🔊 Text to Speech] → [🎧 Preview Audio]
        ↑
  Select engine & voice
```

### Voice Cloning (Coqui)
```
[🎙️ Load Reference Audio] → [🔊 Text to Speech (coqui-tts)] → [🎧 Preview Audio]
```

### With ComfyUI-LLama (Image to Speech)
```
[LoadImage] → [🦙 LLama Server] → [🔊 Text to Speech] → [🎧 Preview Audio]
```

### SRT Subtitle to Audio
```
[📄 Load SRT] → [🔊 Text to Speech] → [💾 Save Audio]
```

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

### Full (with Coqui TTS)
```
pyttsx3>=2.90
pydub>=0.25.1
edge-tts>=6.1.0
TTS>=0.22.0
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

- [edge-tts](https://github.com/rany2/edge-tts) - Microsoft Edge TTS wrapper
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Neural TTS library
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - Offline TTS
