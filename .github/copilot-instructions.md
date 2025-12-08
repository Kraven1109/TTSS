# Copilot / AI Agent Instructions for TTSS

## Project Overview
TTSS is a ComfyUI custom node package for multi-engine text-to-speech synthesis.

**Supported TTS Engines:**
- `pyttsx3` - Offline system voices (SAPI/NSSpeech/espeak)
- `edge-tts` - Microsoft Edge TTS (online, 550+ voices, free)
- `kokoro` - Lightweight neural TTS (82M params, fast, multi-language)
- `orpheus` - SOTA LLM-based TTS with emotion tags (3B params, GPU)
- `csm` - Conversational Speech Model (1B params, conversational, GPU)

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
├── orpheus/             # Orpheus LLM-based TTS models (pre-downloaded via snapshot_download)
├── csm/                 # CSM conversational TTS models (GGUF format)
└── voices/              # Custom voice models
```

## Node Classes (in nodes.py)

| Node | Purpose | Key Method | Engine Support |
|------|---------|------------|----------------|
| `TTSSTextToSpeech` | Main TTS synthesis (with built-in voice selection) | `synthesize()` | All 5 engines |
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
               orpheus_voice, csm_voice, reference_audio, context_audio, ...):
    if engine == "pyttsx3":
        self._synth_pyttsx3(text, output_file, pyttsx3_voice, speed)
    elif engine == "edge-tts":
        self._synth_edge_tts(text, output_file, edge_voice, speed)
    elif engine == "kokoro":
        self._synth_kokoro(text, output_file, kokoro_voice, kokoro_lang, speed)
    elif engine == "orpheus":
        self._synth_orpheus(text, output_file, orpheus_voice)
    elif engine == "csm":
        self._synth_csm(text, output_file, csm_voice, context_audio)
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

### CSM TTS (Conversational Speech Model, 1B params, HuggingFace Transformers)
```python
def _synth_csm(self, text, output_file, speaker_id, context_audio=None):
    import torch
    import torchaudio
    from transformers import CsmForConditionalGeneration, AutoProcessor
    
    # sesame/csm-1b is GATED - requires HuggingFace login:
    #   huggingface-cli login
    #   Then visit https://huggingface.co/sesame/csm-1b and accept terms
    model_id = "sesame/csm-1b"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = CsmForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        dtype=torch.float16,
    )
    
    # Build conversation - role is speaker ID, content is list of typed dicts
    conversation = [
        {"role": f"{speaker_id}", "content": [{"type": "text", "text": text}]}
    ]
    
    # Process and generate
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True
    ).to("cuda")
    
    audio_output = model.generate(**inputs, output_audio=True, max_new_tokens=2048)
    processor.save_audio(audio_output, output_file)  # 24kHz WAV
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
- transformers>=4.52.1 + torchaudio (CSM conversational TTS, requires HuggingFace login)
- huggingface_hub (for model downloading, used by Orpheus pre-download)

## Engine Comparison

| Engine | Params | Quality | Speed | GPU | Voice Clone | Python |
|--------|--------|---------|-------|-----|-------------|--------|
| pyttsx3 | - | ⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | All |
| edge-tts | Cloud | ⭐⭐⭐⭐ | 🚀🚀🚀 | ❌ | ❌ | All |
| kokoro | 82M | ⭐⭐⭐⭐ | 🚀🚀 | Optional | ❌ | 3.10-3.13 |
| orpheus | 3B | ⭐⭐⭐⭐⭐ | 🚀 | Optional | ✅ | 3.10-3.12 |
| csm | 1B | ⭐⭐⭐⭐⭐ | 🚀 | ✅ | ✅ | 3.10+ |

## Why orpheus-cpp instead of orpheus-speech?
- `orpheus-speech` requires vLLM, which doesn't work on Windows natively
- `orpheus-cpp` uses llama.cpp backend - works on Windows/Linux/macOS!
- CPU inference available (slower but no GPU required)
- Same SOTA quality, cross-platform support!

## Why snapshot_download approach like ComfyUI_Qwen3-VL-Instruct?
- **Prevents user cache pollution**: Models download to `ComfyUI/models/tts/orpheus/` instead of `~/.cache/huggingface/`
- **No interference with other extensions**: Temporary environment variables don't affect global ComfyUI session
- **Explicit control**: `snapshot_download` with `local_dir` gives precise control over model storage
- **Follows ComfyUI conventions**: Models stored in proper ComfyUI model directories
- **Reliable caching**: Models stay in ComfyUI directory and don't get accidentally deleted
