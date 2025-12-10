# 🔊 TTSS - Text-to-Speech for ComfyUI

Multi-engine Text-to-Speech nodes for ComfyUI with voice cloning support.

## Features

- **🎤 Multiple TTS Engines**:
  - `pyttsx3` - Offline, uses system voices (Windows SAPI, macOS, Linux espeak)
  - `edge-tts` - Microsoft Edge TTS (online, high quality, free, 550+ voices)
  - `kokoro` - Lightweight neural TTS (82M params, fast, multi-language, Apache 2.0)
  - `orpheus` - SOTA LLM-based TTS with emotion tags (3B params, GPU)

-- **🎙️ Voice Cloning**: (Deprecated) XTTS-v2 (Auralis) voice cloning support removed
- **🌍 Multi-language**: 100+ languages supported
- **😊 Emotion Tags**: Add `<laugh>`, `<sigh>`, `<gasp>` with Orpheus
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
| `TTSSTextToSpeech` | 🔊 | Main TTS node - supports 5 engines with built-in voice selection |
| `TTSSLoadReferenceAudio` | 🎙️ | Load reference audio for voice cloning (deprecated - XTTS-v2 removed) |
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

### 2. edge-tts (Recommended for beginners)
- ✅ High quality Microsoft neural voices
- ✅ 550+ voices in 100+ languages
- ✅ Free, no API key required
- ⚠️ Requires internet connection

**Popular voices:**
- `en-US-AriaNeural` - Female, natural
- `en-US-GuyNeural` - Male, natural
- `vi-VN-HoaiMyNeural` - Vietnamese female
- `ja-JP-NanamiNeural` - Japanese female

### 3. kokoro (Lightweight Neural TTS) 🆕
- ✅ **82M params** - Fast, runs on CPU
- ✅ **Apache 2.0** - Commercial friendly
- ✅ **Multi-language** - EN, ES, FR, JA, ZH, KO, HI, IT, PT
- ✅ **28+ built-in voices** - No reference audio needed
- ✅ **Python 3.10-3.13** - via ONNX Runtime
- ⚠️ First run downloads ~300MB model

**Installation:**
```bash
pip install kokoro-onnx soundfile
```

**Voices:** `af_heart`, `af_bella`, `am_adam`, `bf_emma`, `bm_george`, `af_sarah`...

### 4. orpheus (SOTA LLM-based TTS) 🆕
- ✅ **Human-like speech** - Superior to closed-source models
- ✅ **Emotion tags** - `<laugh>`, `<sigh>`, `<gasp>`, `<chuckle>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`
- ✅ **Zero-shot voice cloning**
- ✅ **~200ms latency** - Real-time streaming
- ✅ **Works on Windows/Linux/macOS** - via llama.cpp backend
- ⚠️ **Python 3.10-3.12 required** for pre-built wheels
- ⚠️ First run downloads ~3GB GGUF model

**Installation (Python 3.10-3.12 only):**
```bash
pip install orpheus-cpp
# CPU inference:
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# GPU acceleration (CUDA 12.1-12.5):
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

> ⚠️ **Python 3.13**: No pre-built wheels. Requires building llama-cpp-python from source with CUDA toolkit.

**Voices:** `tara`, `leah`, `jess`, `leo`, `dan`, `mia`, `zac`, `zoe`

**Emotion tags example:**
```
I can't believe it! <laugh> This is amazing <gasp>
```

### Orpheus emotive control: inline tags vs descriptors

Orpheus supports two primary ways to control emotion and expressive behavior:

- Inline emotive tags (e.g., `<laugh>`) — these are non-verbal or short expressive events inserted directly into text and are processed by the model at generation time.
- Emotion descriptors & modifiers (e.g., `happy`, `sad`, `whisper`, `fast`) — these are free-form descriptors or tokens used in the prompt that bias the model's prosody/intonation across a broader span of text. Descriptor support varies by model.

Supported inline emotive tags (English models):

| Tag | When to use | Example |
|-----|-------------|--------|
| `<laugh>` | Laughter sound | "I can't believe you did that! `<laugh>` That is hilarious." |
| `<chuckle>` | Short chuckle | "He thinks he can beat me? `<chuckle>` Good luck with that." |
| `<sigh>` | Relief, tiredness, resignation | "`<sigh>` Finally, it's done." |
| `<cough>` | Throat clear/interruption | "Ahem `<cough>` excuse me..." |
| `<sniffle>` | Sad, tearful sniff | "It's just so hard `<sniffle>`" |
| `<groan>` | Disgust/annoyance | "Not again `<groan>`..." |
| `<yawn>` | Bored/ tired | "I'm so tired `<yawn>`" |
| `<gasp>` | Sudden surprise | "What?! `<gasp>` Are you serious?" |

Emotion Descriptors & Modifiers (used in prompts; not inline tags):

| Type | Examples |
|------|----------|
| Core emotions | `happy`, `normal`, `sad`, `angry`, `excited`, `surprised` |
| Intensity / style modifiers | `whisper`, `shout`, `deep`, `high`, `slow`, `fast` |
| Complex/other states | `frustrated`, `panicky`, `curious`, `sleepy`, `crying` |

> Important: Descriptor support varies per model (depends on training & fine-tuning). The inline tags above are supported in the base English Orpheus models per the official Orpheus docs; descriptors are more model-dependent. Always test changes in short segments per voice/model.

### 5. xtts-v2 (Voice Cloning via Auralis) — Deprecated
- This integration has been deprecated and removed from the TTSS plugin. Use Orpheus/CSM capabilities where applicable for voice shaping.

## Directory Structure

```
ComfyUI/
├── models/
│   └── tts/                    # TTS model directory
│       ├── reference_audio/    # (Deprecated) Voice cloning reference files (.wav, 6+ seconds)
│       # Note: xtts/ models and voice cloning support removed (XTTS-v2 / Auralis)
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

### Voice Cloning (XTTS-v2) — Deprecated
```
Stationary note: XTTS-v2 and Auralis-based voice cloning are no longer supported in this plugin. Keep your reference_audio files for external use or future features.
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
-- `engine` (dropdown) - pyttsx3 / edge-tts / kokoro / orpheus / csm
- `speed` (FLOAT) - 0.5 to 2.0
- `edge_voice` (dropdown) - 550+ Microsoft Edge voices
- `pyttsx3_voice` (dropdown) - System voices
- `kokoro_voice` (dropdown) - Kokoro built-in voices (28+)
- `kokoro_lang` (dropdown) - Language code (a=EN, j=JA, z=ZH...)
-- `orpheus_voice` (dropdown) - Orpheus voices (tara, leo, mia...)
- `text_input` (STRING, optional) - Piped text input
- `srt_input` (SRT, optional) - SRT file path
-- `reference_audio` (AUDIOPATH, optional) - (Deprecated) Not used in the current plugin
 - `show_orpheus_help` (BOOL, optional) - Show Orpheus inline tags & descriptors info in logs and warnings.

**Outputs:**
- `audio_path` (AUDIOPATH)

### TTSConversation 💬
**Inputs:**
- `conversation_text` (STRING) - Conversation text with lines prefixed by [0], [1], etc.
- `speaker_0_voice`.. `speaker_3_voice` (STRING) - Speaker IDs for CSM
- `context_audio` (AUDIOPATH, optional) - Optional context audio to preserve continuity
- `max_speakers` (INT, optional) - Maximum speakers to process

**Outputs:**
- `conversation_audio` (AUDIOPATH)

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
soundfile>=0.12.0
```

### With Kokoro (lightweight neural TTS)
```
kokoro>=0.9.4
soundfile>=0.12.0
# Also install espeak-ng system package
```

### With Orpheus (SOTA LLM TTS, requires GPU)
```
orpheus-speech
vllm==0.7.3
```

### XTTS-v2 (Deprecated)
This plugin no longer supports XTTS-v2 (Auralis) voice cloning. The following requirements are for historical reference only:
```
auralis>=0.2.8
torch>=2.0.0
```

**Note:** pydub requires ffmpeg. Install via:
- Windows: `conda install ffmpeg` or download from ffmpeg.org
- Linux: `apt install ffmpeg`
- macOS: `brew install ffmpeg`

## Engine Comparison

| Engine | Quality | Speed | GPU | Voice Clone | Multi-lang | Python |
|--------|---------|-------|-----|-------------|------------|--------|
| pyttsx3 | ⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | Limited | All |
| edge-tts | ⭐⭐⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | 100+ | All |
| kokoro | ⭐⭐⭐⭐ | 🚀🚀 | Optional | ❌ | 9 | 3.9+ |
| orpheus | ⭐⭐⭐⭐⭐ | 🚀 | ✅ Required | ✅ | 7 | 3.9+ |
| xtts-v2 | Deprecated | - | - | - | - | - |

## Related Projects

### CSM / unsloth/csm-1b (Conversational Speech Model)

- Default CSM model used by TTSS: `unsloth/csm-1b` (Apache-2.0 - public).
- CSM offers high-quality conversational TTS with multi-speaker context. It requires torch/torchaudio and a suitable GPU for best performance.

- **[ComfyUI-LLama](https://github.com/Kraven1109/ComfyUI-Llama)** - LLM inference for image description → TTS pipeline

## License

MIT License

## Acknowledgements

- [Kokoro](https://github.com/hexgrad/kokoro) - Lightweight 82M neural TTS
- [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) - SOTA LLM-based TTS
- [Auralis](https://github.com/astramind-ai/Auralis) - XTTS-v2 wrapper (Deprecated / historical)
- [edge-tts](https://github.com/rany2/edge-tts) - Microsoft Edge TTS wrapper
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) - Offline TTS
