# Copilot / AI Agent Instructions for TTSS

## Project Overview
TTSS is a ComfyUI custom node package for multi-engine text-to-speech synthesis.

**Supported TTS Engines:**
- `pyttsx3` - Offline system voices (SAPI/NSSpeech/espeak)
- `edge-tts` - Microsoft Edge TTS (online, 550+ voices, free)
- `kokoro` - Lightweight neural TTS (82M params, fast, multi-language)
- `orpheus` - SOTA LLM-based TTS with emotion tags (3B params, GPU)

## Project Structure
```
comfyUI-TTSS/
├── __init__.py          # ComfyUI node registration (NODE_CLASS_MAPPINGS)
├── nodes.py             # All node implementations (7 nodes)
├── requirements.txt     # Python dependencies
├── web/ttss.js          # Frontend audio preview widget
├── examples/            # Example ComfyUI workflows
└── .github/             # GitHub configs & Copilot instructions
```

## Model Directory (ComfyUI Convention)
```
ComfyUI/models/tts/
├── kokoro/              # Kokoro ONNX models
├── reference_audio/     # Voice cloning reference files (.wav, 6+ seconds)
├── orpheus/             # Orpheus LLM-based TTS models
└── voices/              # Custom voice models
```

## Node Classes (in nodes.py)

| Node | Purpose | Key Method | Engine Support |
|------|---------|------------|----------------|
| `TTSSTextToSpeech` | Main TTS synthesis (with built-in voice selection) | `synthesize()` | All 4 engines |
| `TTSSLoadAudio` | Load audio files | `load_audio()` | - |
| `TTSSLoadSRT` | Load SRT subtitles | `load_srt()` | - |
| `TTSSPreviewAudio` | Audio preview in UI | `preview()` | - |
| `TTSSCombineAudio` | Merge audio files | `combine()` | - |
| `TTSSSaveAudio` | Save with format | `save_audio()` | - |

## Key Code Patterns

### Multi-Engine TTS Pattern
```python
def synthesize(self, text, engine, speed, 
               pyttsx3_voice, edge_voice, kokoro_voice, kokoro_lang,
               orpheus_voice, reference_audio, ...):
    if engine == "pyttsx3":
        self._synth_pyttsx3(text, output_file, pyttsx3_voice, speed)
    elif engine == "edge-tts":
        self._synth_edge_tts(text, output_file, edge_voice, speed)
    elif engine == "kokoro":
        self._synth_kokoro(text, output_file, kokoro_voice, kokoro_lang, speed)
    elif engine == "orpheus":
        self._synth_orpheus(text, output_file, orpheus_voice)
```

### Kokoro TTS (82M params, ONNX Runtime, Python 3.10-3.13)
```python
def _synth_kokoro(self, text, output_file, voice, lang_code, speed):
    from kokoro_onnx import Kokoro
    import soundfile as sf
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    sf.write(output_file, samples, sample_rate)
```

### Orpheus TTS (SOTA LLM-based with emotion tags, llama.cpp backend)
```python
def _synth_orpheus(self, text, output_file, voice):
    from orpheus_cpp import OrpheusCpp
    import numpy as np
    from scipy.io.wavfile import write as wav_write
    import os
    
    # Set model cache to ComfyUI models directory
    orpheus_models_path = os.path.join(tts_models_path, "orpheus")
    os.makedirs(orpheus_models_path, exist_ok=True)
    os.environ["HF_HOME"] = orpheus_models_path
    
    # Suppress ONNX Runtime CUDA warnings for better UX
    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "2"  # Warning level only
    os.environ["ORT_DISABLE_CUDA_GRAPH"] = "1"  # Disable CUDA graph warnings
    
    orpheus = OrpheusCpp(verbose=False, lang="en", n_gpu_layers=-1)  # GPU enabled
    buffer = []
    sample_rate = None
    for i, (sr, chunk) in enumerate(orpheus.stream_tts_sync(text, options={"voice_id": voice})):
        if sample_rate is None:
            sample_rate = sr
        # Ensure chunk is 1D for concatenation
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        buffer.append(chunk)
    audio = np.concatenate(buffer, axis=0)  # Concatenate along time axis
    wav_write(output_file, sample_rate, audio)
    # Supports emotion tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
    # Additional expressive tags: <happy>, <normal>, <disgust>, <sad>, <frustrated>, <slow>, <excited>, <whisper>, <panicky>, <curious>, <surprise>, <fast>, <crying>, <deep>, <sleepy>, <angry>, <high>, <shout>
```

### Dynamic Voice Loading
```python
def get_edge_tts_voices():
    """Cached voice list from edge-tts CLI"""
    cache_file = os.path.join(tts_models_path, "edge_voices_cache.json")
    # Returns 550+ voices, cached for 24h
```

### ComfyUI Node Pattern
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_name": ("TYPE", {"default": value})}}
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    FUNCTION = "method_name"
    CATEGORY = "🔊 TTSS"
    
    def method_name(self, input_name):
        return (result,)
```

## Developer Workflows

### Run Python files (ComfyUI embedded Python)
```powershell
d:\Apps\ComfyUI_portable\python_embeded\python.exe <file>.py
```

### Install dependencies
```powershell
d:\Apps\ComfyUI_portable\python_embeded\python.exe -m pip install -r requirements.txt
```

### Test node import
```powershell
d:\Apps\ComfyUI_portable\python_embeded\python.exe -c "import sys; sys.path.insert(0, 'D:/Apps/ComfyUI_portable/ComfyUI'); import folder_paths; from nodes import TTSSTextToSpeech; print('OK')"
```

## Integration with ComfyUI-LLama
TTSS works with ComfyUI-LLama for image-to-speech:
```
LoadImage → 🦙 LLama Server → 🔊 Text to Speech → 🎧 Preview Audio
```

## Editing Rules

1. Keep all node logic in `nodes.py`, registration in `__init__.py`
2. Use `folder_paths.models_dir` for model paths (ComfyUI convention)
3. Category prefix: `"🔊 TTSS"` for all nodes
4. Display name prefix: `"TTSS "` for searchability
5. Return tuples from node methods: `return (result,)`
6. Cache voice lists to avoid repeated API calls
7. Support graceful fallback when optional engines not installed

## Dependencies

**Required:**
- pyttsx3 (offline TTS)
- pydub (audio processing)
- srt (subtitle parsing)

**Optional:**
- edge-tts (Microsoft TTS, 550+ voices)
- kokoro-onnx (lightweight neural TTS, 82M params, Python 3.10-3.13)
- orpheus-cpp + llama-cpp-python (SOTA LLM TTS, llama.cpp backend, works on Windows!)

## Engine Comparison

| Engine | Params | Quality | Speed | GPU | Voice Clone | Python |
|--------|--------|---------|-------|-----|-------------|--------|
| pyttsx3 | - | ⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | All |
| edge-tts | Cloud | ⭐⭐⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | All |
| kokoro | 82M | ⭐⭐⭐⭐ | 🚀🚀 | Optional | ❌ | 3.10-3.13 |
| orpheus | 3B | ⭐⭐⭐⭐⭐ | 🚀 | Optional | ✅ | 3.10-3.12 |

## Why orpheus-cpp instead of orpheus-speech?
- `orpheus-speech` requires vLLM, which doesn't work on Windows natively
- `orpheus-cpp` uses llama.cpp backend - works on Windows/Linux/macOS!
- CPU inference available (slower but no GPU required)
- Same SOTA quality, cross-platform support!
