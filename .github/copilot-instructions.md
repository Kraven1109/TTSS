# Copilot / AI Agent Instructions for TTSS

## Project Overview
TTSS is a ComfyUI custom node package for multi-engine text-to-speech synthesis.

**Supported TTS Engines:**
- `pyttsx3` - Offline system voices (SAPI/NSSpeech/espeak)
- `edge-tts` - Microsoft Edge TTS (online, 550+ voices, free)
- `kokoro` - Lightweight neural TTS (82M params, fast, multi-language)
- `orpheus` - SOTA LLM-based TTS with emotion tags and multilingual support (3B params, GPU)
- `csm` - Conversational Speech Model (1B params, conversational, GPU)

## Project Structure
```
comfyUI-TTSS/
├── __init__.py          # ComfyUI node registration (NODE_CLASS_MAPPINGS)
├── nodes.py             # All node implementations (8 nodes)
├── requirements.txt     # Python dependencies
├── web/ttss.js          # Frontend audio preview widget
├── examples/            # Example ComfyUI workflows
└── .github/             # GitHub configs & Copilot instructions
```

## Model Directory (ComfyUI Convention)
```
ComfyUI/models/tts/
├── kokoro/              # Kokoro ONNX models (auto-downloaded via HuggingFace Hub)
├── reference_audio/     # Voice cloning reference files (.wav, 6+ seconds)
├── orpheus/             # Orpheus LLM-based TTS models (pre-downloaded via snapshot_download)
├── csm/                 # CSM conversational TTS models (pre-downloaded via snapshot_download)
└── voices/              # Custom voice models
```

## Node Classes (in nodes.py)

| Node | Purpose | Key Method | Engine Support |
|------|---------|------------|----------------|
| `TTSSTextToSpeech` | Main TTS synthesis (with built-in voice selection) | `synthesize()` | All 5 engines |
| `TTSConversation` | Multi-speaker conversational TTS | `generate_conversation()` | CSM only |
| `TTSSLoadAudio` | Load audio files | `load_audio()` | - |
| `TTSSLoadSRT` | Load SRT subtitles | `load_srt()` | - |
| `TTSSPreviewAudio` | Audio preview in UI | `preview()` | - |
| `TTSSCombineAudio` | Merge audio files | `combine()` | - |
| `TTSSSaveAudio` | Save with format | `save_audio()` | - |

## Voice Definitions

### Orpheus Voices (Multilingual - 24 voices across 8 languages)
```python
ORPHEUS_VOICES = {
    "English": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    "French": ["fr_speaker_0", "fr_speaker_1", "fr_speaker_2"],
    "German": ["de_speaker_0", "de_speaker_1", "de_speaker_2"],
    "Korean": ["ko_speaker_0", "ko_speaker_1", "ko_speaker_2"],
    "Hindi": ["hi_speaker_0", "hi_speaker_1", "hi_speaker_2"],
    "Mandarin": ["zh_speaker_0", "zh_speaker_1", "zh_speaker_2"],
    "Spanish": ["es_speaker_0", "es_speaker_1", "es_speaker_2"],
    "Italian": ["it_speaker_0", "it_speaker_1", "it_speaker_2"]
}
```

### CSM Voices (Speaker IDs for conversational TTS)
```python
CSM_VOICES = [str(i) for i in range(10)]  # 0-9 speaker IDs
```

## Key Code Patterns

### Multi-Engine TTS Pattern
```python
def synthesize(self, text, engine, speed, 
               pyttsx3_voice, edge_voice, kokoro_voice, kokoro_lang,
               orpheus_voice, csm_voice, context_audio, ...):
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
    
    # Auto-download models to ComfyUI directory
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    
    # Long-form processing: Handle phoneme limit (~510) by chunking text
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Chunk text to stay under phoneme limits
    max_phonemes_per_chunk = 400  # Conservative limit
    text_chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        estimated_phonemes = len(current_chunk + sentence) * 1.5
        if estimated_phonemes < max_phonemes_per_chunk:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                text_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        text_chunks.append(current_chunk.strip())
    
    # Process chunks and concatenate
    all_audio_chunks = []
    for chunk_text in text_chunks:
        samples, sample_rate = kokoro.create(chunk_text, voice=voice, speed=speed, lang="en-us")
        all_audio_chunks.append(samples)
    
    final_audio = np.concatenate(all_audio_chunks, axis=0)
    sf.write(output_file, final_audio, sample_rate)
```

### CSM TTS (Conversational Speech Model, 1B params, HuggingFace Transformers)
```python
def _synth_csm(self, text, output_file, speaker_id, context_audio=None):
    import torch
    import torchaudio
    from transformers import CsmForConditionalGeneration, AutoProcessor
    
    # By default, TTSS uses `unsloth/csm-1b` (Apache-2.0). If you prefer `sesame/csm-1b`,
    # it is gated and requires a HuggingFace login and acceptance of the terms on their page.
    model_id = "unsloth/csm-1b"
    
    # Download to ComfyUI models directory (not user cache)
    local_model_path = os.path.join(tts_csm_path, model_id.replace("/", "_"))
    if not os.path.exists(local_model_path):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
        )
    
    processor = AutoProcessor.from_pretrained(local_model_path)
    model = CsmForConditionalGeneration.from_pretrained(
        local_model_path,
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

### Orpheus TTS (3B params, llama.cpp backend, Multi-language)
```python
def _synth_orpheus(self, text, output_file, lang, voice):
    """Enhanced Orpheus with multilingual support and long-form processing."""
    from orpheus_cpp import OrpheusCpp
    import numpy as np
    from scipy.io.wavfile import write as wav_write
    
    # Long-form processing: Split text into sentences for better quality
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into chunks (~300 chars per chunk)
    text_chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) < 300:
            current_chunk += sentence + " "
        else:
            text_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        text_chunks.append(current_chunk.strip())
    
    # Initialize with correct language
    lang_code = lang[:2].lower()  # "en", "fr", "de", etc.
    orpheus = OrpheusCpp(verbose=False, lang=lang_code, n_gpu_layers=-1)
    
    # Process chunks with crossfade stitching
    all_audio_chunks = []
    for chunk_text in text_chunks:
        # Generate audio for chunk
        chunk_buffer = []
        for sr, audio_chunk in orpheus.stream_tts_sync(chunk_text, options={"voice_id": voice}):
            chunk_buffer.append(audio_chunk)
        chunk_audio = np.concatenate(chunk_buffer, axis=0)
        all_audio_chunks.append(chunk_audio)
    
    # Crossfade stitching (200ms at 24kHz)
    if len(all_audio_chunks) > 1:
        crossfade_samples = int(0.2 * sr)
        for i in range(1, len(all_audio_chunks)):
            # Apply linear crossfade
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)
            # Mix overlapping regions
            mixed = all_audio_chunks[i-1][-crossfade_samples:] * fade_out + \
                   all_audio_chunks[i][:crossfade_samples] * fade_in
            # Combine: prev - overlap + mixed + curr - overlap
            all_audio_chunks[i-1] = np.concatenate([
                all_audio_chunks[i-1][:-crossfade_samples], mixed, 
                all_audio_chunks[i][crossfade_samples:]
            ])
            all_audio_chunks[i] = all_audio_chunks[i-1]
    
    # Save final audio
    wav_write(output_file, sr, all_audio_chunks[-1])
```

### CSM Emotional / Expressive Tags
CSM supports emotional and expressive control through text prompts. While not fully documented, the model can interpret tags for:

**Emotion:** happy, sad, angry, empathetic, excited, calm, warm, cold, harsh, soft

**Style:** formal, casual, storytelling, dramatic, energetic, reassuring

**Expressiveness:** whispering, shouting, enthusiastic, thoughtful, confident

**Examples:**
- `[0]I'm so excited to meet you!` - enthusiastic tone
- `[1]I understand how you feel.` - empathetic, warm tone  
- `[2]This is unacceptable!` - angry, harsh tone
- `[3]Let me tell you a story...` - storytelling, dramatic style

**Note:** Emotional control is achieved through natural language prompts rather than explicit tags. The model learns emotional context from conversational patterns.

### HuggingFace Token Permissions for Gated Models
Even if you've run `huggingface-cli login`, your access token may not have the right permissions for gated models like `sesame/csm-1b`.

**Problem:** By default, fine-grained tokens can be restricted (e.g., only read/write to certain repos). If "Access to public gated repos" is not enabled, you'll get a 403 error.

**Solution:**
1. Go to [Hugging Face Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Edit or create a new token
3. Make sure **"Access to gated repositories"** is enabled
4. The token needs this permission to access models that require agreement acceptance

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
