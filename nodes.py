"""
TTSS ComfyUI Nodes
Text-to-Speech nodes with multiple engine support.
Supports: pyttsx3 (offline), edge-tts (Microsoft), Coqui TTS (neural).
"""

# =============================================================================
# Environment Variables (MUST be set BEFORE any imports to take effect)
# =============================================================================
import os

# Suppress ONNX Runtime CUDA warnings (must be set before any ONNX import)
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  # Error level only (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)

import time
import json
import hashlib
import subprocess
import threading
import shutil
import tempfile

# Import TTS engine functions
try:
    from . import engines
except ImportError:
    # Fallback for direct import (not as package)
    import engines

try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    class FolderPaths:
        @staticmethod
        def get_input_directory():
            return os.path.join(os.path.dirname(__file__), "input")
        @staticmethod
        def get_output_directory():
            return os.path.join(os.path.dirname(__file__), "output")
        models_dir = os.path.join(os.path.dirname(__file__), "models")
    folder_paths = FolderPaths()

# =============================================================================
# Paths Setup
# =============================================================================
now_dir = os.path.dirname(os.path.abspath(__file__))
input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()

# TTS Models directory - follows ComfyUI convention
if COMFYUI_AVAILABLE:
    tts_models_path = os.path.join(folder_paths.models_dir, "tts")
else:
    tts_models_path = os.path.join(now_dir, "models", "tts")

# Create subdirectories
tts_reference_path = os.path.join(tts_models_path, "reference_audio")
tts_voices_path = os.path.join(tts_models_path, "voices")
tts_orpheus_path = os.path.join(tts_models_path, "orpheus")  # For Orpheus LLM-based TTS
tts_csm_path = os.path.join(tts_models_path, "csm")  # For CSM conversational TTS

for path in [output_path, tts_models_path, tts_reference_path, tts_voices_path, tts_orpheus_path, tts_csm_path]:
    os.makedirs(path, exist_ok=True)

# Note: HF environment variables are set temporarily during Orpheus usage to avoid
# interfering with other ComfyUI nodes that use HuggingFace Hub

# =============================================================================
# TTS Engine Registry
# =============================================================================
TTS_ENGINES = ["pyttsx3", "edge-tts", "kokoro", "orpheus", "csm"]

# Kokoro voices (built-in, no reference audio needed)
KOKORO_VOICES = [
    # American English
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    # British English  
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]

# Kokoro language codes
KOKORO_LANGS = {
    "a": "American English",
    "b": "British English", 
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}

# Orpheus voices (multilingual - 24 voices across 8 languages)
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

# Orpheus inline emotive tags (supported tags per Orpheus-TTS README)
# These are processed directly in prompt text and produce non-verbal or short expressives in audio.
# The supported inline tags (English models) per Orpheus docs are:
ORPHEUS_EMOTIONS = [
    "<laugh>",
    "<chuckle>",
    "<sigh>",
    "<cough>",
    "<sniffle>",
    "<groan>",
    "<yawn>",
    "<gasp>",
]

# Orpheus emotion descriptors and modifiers (text descriptors that influence style)
# These are model descriptors used in prompts (not necessarily inline tags) and may vary by model.
ORPHEUS_DESCRIPTORS = [
    "happy", "normal", "sad", "angry", "excited", "surprised",
    # Intensity/style modifiers
    "whisper", "shout", "deep", "high", "slow", "fast",
    # More complex states
    "frustrated", "panicky", "curious", "sleepy", "crying",
    # Speech traits
    "longer", "disgust",
]

# CSM voices (speaker IDs for conversational TTS)
CSM_VOICES = [str(i) for i in range(10)]  # 0-9 speaker IDs

# Model repository IDs for auto-download
KOKORO_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
CSM_MODEL_ID = "unsloth/csm-1b"
SNAC_REPO_ID = "hubertsiuzdak/snac_24khz"

# Orpheus model - unified model for all languages via Transformers
ORPHEUS_MODEL_ID = "unsloth/orpheus-3b-0.1-ft"


# =============================================================================
# Model Manager: Cache & Unload helpers for one-shot workflows
# =============================================================================
class ModelManager:
    """Manage loaded model instances with unload helpers.
    This keeps models if requested but can unload after use to free GPU/RAM.
    """
    def __init__(self):
        self._cache = {}
        self._locks = {}

    def _lock(self, key):
        if key not in self._locks:
            self._locks[key] = threading.Lock()
        return self._locks[key]

    def get_kokoro(self, model_path: str, voices_path: str):
        key = f"kokoro:{model_path}:{voices_path}"
        with self._lock(key):
            if key not in self._cache:
                try:
                    from kokoro_onnx import Kokoro
                except Exception:
                    raise
                self._cache[key] = Kokoro(model_path, voices_path)
            return self._cache[key]

    def unload_kokoro(self, model_path: str, voices_path: str):
        key = f"kokoro:{model_path}:{voices_path}"
        with self._lock(key):
            if key in self._cache:
                try:
                    del self._cache[key]
                except Exception:
                    pass

    def get_csm(self, local_model_path: str, device: str = "cuda"):
        key = f"csm:{local_model_path}:{device}"
        with self._lock(key):
            if key not in self._cache:
                try:
                    import torch
                    from transformers import CsmForConditionalGeneration, AutoProcessor
                except Exception:
                    raise
                processor = AutoProcessor.from_pretrained(local_model_path)
                model = CsmForConditionalGeneration.from_pretrained(
                    local_model_path,
                    device_map=device,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                )
                self._cache[key] = (processor, model)
            return self._cache[key]

    def unload_csm(self, local_model_path: str, device: str = "cuda"):
        key = f"csm:{local_model_path}:{device}"
        with self._lock(key):
            if key in self._cache:
                try:
                    _, model = self._cache[key]
                    model.to('cpu')
                    del model
                except Exception:
                    pass
                del self._cache[key]

                # Force memory cleanup
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    def clear_all(self):
        """Clear all cached models and attempt to free resources."""
        keys = list(self._cache.keys())
        for k in keys:
            try:
                del self._cache[k]
            except Exception:
                pass

        # Force memory cleanup
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

# Global ModelManager for this nodes module
MODEL_MANAGER = ModelManager()


def get_pyttsx3_voices():
    """Get available pyttsx3 system voices."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        voice_list = []
        for v in voices:
            # Clean up voice name
            name = v.name.split(" - ")[0] if " - " in v.name else v.name
            voice_list.append(name)
        engine.stop()
        return voice_list if voice_list else ["default"]
    except Exception as e:
        print(f"[TTSS] Error getting pyttsx3 voices: {e}")
        return ["default"]

def get_edge_tts_voices():
    """Get available edge-tts voices (cached)."""
    cache_file = os.path.join(tts_models_path, "edge_voices_cache.json")
    
    # Return cached if exists and fresh (24h)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if time.time() - cache.get('timestamp', 0) < 86400:
                return cache.get('voices', ["en-US-AriaNeural"])
        except:
            pass
    
    # Try to fetch using edge-tts CLI
    try:
        edge_tts_cmd = _get_edge_tts_cli()
        result = subprocess.run(
            [edge_tts_cmd, '--list-voices'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            voices = []
            lines = result.stdout.strip().split('\n')
            # Skip header lines (Name, ----)
            for line in lines[2:]:  # Skip first 2 header lines
                if line.strip():
                    # First column is voice name (e.g., "af-ZA-AdriNeural")
                    parts = line.split()
                    if parts and 'Neural' in parts[0]:
                        voices.append(parts[0])
            
            if voices:
                # Cache voices
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'timestamp': time.time(), 'voices': voices}, f)
                return voices
    except Exception as e:
        print(f"[TTSS] Error getting edge-tts voices via CLI: {e}")
    
    # Fallback to common voices
    return [
        "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
        "en-GB-SoniaNeural", "en-AU-NatashaNeural",
        "vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural",
        "ja-JP-NanamiNeural", "ko-KR-SunHiNeural",
        "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
        ]

def get_kokoro_voices():
    """Get available Kokoro voices."""
    return KOKORO_VOICES

def get_kokoro_langs():
    """Get Kokoro language codes."""
    return list(KOKORO_LANGS.keys())

def get_orpheus_voices():
    """Get available Orpheus voices (flattened list for dropdown)."""
    all_voices = []
    for lang, voices in ORPHEUS_VOICES.items():
        prefix = lang[:2].lower()
        for voice in voices:
            if voice.startswith(f"{prefix}_"):
                all_voices.append(voice)
            else:
                all_voices.append(f"{prefix}_{voice}")  # e.g., "en_tara", "fr_speaker_0"
    return all_voices

def get_orpheus_languages():
    """Get available Orpheus languages."""
    return list(ORPHEUS_VOICES.keys())

def get_csm_voices():
    """Get available CSM speaker IDs."""
    return CSM_VOICES

# NOTE: XTTS-v2 (Auralis voice cloning) removed from supported engines
def get_reference_audio_files():
    """No-op: reference audio is no longer used for voice cloning (XTTS removed).
    Kept for backward compatibility in examples but returns only "(none)".
    """
    return ["(none)"]


def detect_inline_tags(text: str):
    """Detect inline <tag> occurrences in the text and return set of tag names without brackets."""
    import re
    return set(re.findall(r'<([a-zA-Z0-9_]+)>', text))


def validate_orpheus_tag_usage(engine: str, text: str):
    """Validator to warn about misuse of Orpheus inline tags and descriptor-in-tag misuse.

    - Warn when inline tags are used with non-Orpheus engines.
    - Warn when an inline tag contains a descriptor (e.g., <happy>) but descriptors aren't inline tags.
    """
    tags = detect_inline_tags(text)
    if not tags:
        return

    # If engine is not Orpheus, warn that inline tags will be removed/ignored.
    if engine != 'orpheus':
        print(f"[TTSS] Warning: Inline emotive tags {sorted(['<' + t + '>' for t in tags])} detected but engine '{engine}' does not support them; they will be removed.")
        return

    # engine == 'orpheus': check whether any tag is actually a descriptor
    bad_as_inline = []
    for tag in tags:
        bracketed = f"<{tag}>"
        if bracketed not in ORPHEUS_EMOTIONS:
            # If it's in descriptors, this is likely an incorrect inline usage (e.g., <happy>)
            if tag in ORPHEUS_DESCRIPTORS:
                bad_as_inline.append(tag)
    if bad_as_inline:
        print(f"[TTSS] Warning: The following descriptors are used as inline tags: {', '.join(['<' + t + '>' for t in bad_as_inline])}. Descriptors should be used as prompt text (e.g., 'happy'), not inside angle brackets.")


# =============================================================================
# Node: TTSSTextToSpeech (Multi-Engine with integrated voice selection)
# =============================================================================
class TTSSTextToSpeech:
    """
    Text-to-Speech synthesis with multiple engine support.
    - pyttsx3: Offline, uses system voices (SAPI/NSSpeech/espeak)
    - edge-tts: Microsoft Edge TTS (online, high quality, free, 550+ voices)
    - kokoro: Lightweight neural TTS (82M params, fast, multi-language)
    - orpheus: SOTA LLM-based TTS with emotion tags (3B params, GPU)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get voices for each engine
        pyttsx3_voices = get_pyttsx3_voices()
        edge_voices = get_edge_tts_voices()
        kokoro_voices = get_kokoro_voices()
        orpheus_voices = get_orpheus_voices()
        csm_voices = get_csm_voices()
        
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Hello, this is a test of text to speech.",
                    "multiline": True
                }),
                "engine": (TTS_ENGINES, {"default": "edge-tts"}),
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1
                }),
                "keep_models": ("BOOL", {"default": False, "tooltip": "Keep models in memory after synthesis for faster subsequent runs (uses more RAM/GPU)"}),
                "show_orpheus_help": ("BOOL", {"default": False, "tooltip": "Show Orpheus inline tags & descriptors in logs and warnings"}),
                # Voice dropdowns for each engine
                "edge_voice": (edge_voices, {"default": "en-US-AriaNeural"}),
                "pyttsx3_voice": (pyttsx3_voices, {"default": pyttsx3_voices[0] if pyttsx3_voices else "default"}),
                "kokoro_voice": (kokoro_voices, {"default": "af_heart"}),
                "kokoro_lang": (list(KOKORO_LANGS.keys()), {"default": "a"}),
                "orpheus_lang": (get_orpheus_languages(), {"default": "English"}),
                "orpheus_voice": (orpheus_voices, {"default": "en_tara", "tooltip": "Orpheus inline tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>. Descriptors (e.g., 'happy') are model-dependent and should not be used inside <...>.", }),
                "csm_voice": (csm_voices, {"default": "0"}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True, "multiline": True}),
                "srt_input": ("SRT",),
                "context_audio": ("AUDIOPATH",),  # For CSM conversational context
            }
        }
    
    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "synthesize"
    CATEGORY = "🔊 TTSS"
    
    def synthesize(self, text, engine, speed, 
                   pyttsx3_voice="default", edge_voice="en-US-AriaNeural",
                   kokoro_voice="af_heart", kokoro_lang="a",
                   orpheus_lang="English", orpheus_voice="en_tara", csm_voice="0",
                   text_input=None, srt_input=None, context_audio=None, show_orpheus_help=False, keep_models=False):
        """Generate speech from text using selected engine."""
        
        # Build final text
        final_text = self._build_text(text, text_input, srt_input)
        if not final_text:
            raise ValueError("[TTSS] No text provided for synthesis")
        
        # Validate Orpheus inline tag usage and provide helpful warnings
        try:
            validate_orpheus_tag_usage(engine, final_text)
        except Exception as _e:
            # Validator should not cause failures; log and continue
            print(f"[TTSS] Validator error: {_e}")

        if show_orpheus_help:
            print("[TTSS] Orpheus inline tags:\n  ", ", ".join(ORPHEUS_EMOTIONS))
            print("[TTSS] Orpheus descriptors:\n  ", ", ".join(ORPHEUS_DESCRIPTORS))
        
        # Select voice/model based on engine
        if engine == "pyttsx3":
            voice_name = pyttsx3_voice
        elif engine == "edge-tts":
            voice_name = edge_voice
        elif engine == "kokoro":
            voice_name = kokoro_voice
        elif engine == "orpheus":
            voice_name = orpheus_voice
        elif engine == "csm":
            voice_name = csm_voice
        else:
            voice_name = ""
        
        print(f"[TTSS] Engine: {engine}, Voice: {voice_name}, Text: {final_text[:80]}...")
        
        # Generate unique filename based on content
        text_hash = hashlib.md5(f"{final_text}{engine}{voice_name}{speed}".encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_file = os.path.join(output_path, f"ttss_{engine}_{timestamp}_{text_hash}.wav")
        
        
        # Route to appropriate engine
        try:
            if engine == "pyttsx3":
                self._synth_pyttsx3(final_text, output_file, voice_name, speed)
            elif engine == "edge-tts":
                self._synth_edge_tts(final_text, output_file, voice_name, speed)
            elif engine == "kokoro":
                self._synth_kokoro(final_text, output_file, voice_name, kokoro_lang, speed, keep_models)
            elif engine == "orpheus":
                self._synth_orpheus(final_text, output_file, orpheus_lang, voice_name, keep_models)
            elif engine == "csm":
                self._synth_csm(final_text, output_file, voice_name, context_audio)
            else:
                raise ValueError(f"[TTSS] Unknown engine: {engine}")
        except Exception as synth_err:
            raise RuntimeError(f"[TTSS] Engine '{engine}' failed: {synth_err}") from synth_err
        
        if not os.path.exists(output_file):
            raise RuntimeError(f"[TTSS] Failed to create audio: {output_file}")
        
        print(f"[TTSS] Audio saved: {output_file}")
        return (output_file,)
    
    def _build_text(self, text, text_input, srt_input):
        """Combine text inputs."""
        parts = []
        if text and text.strip():
            parts.append(text.strip())
        if text_input and text_input.strip():
            parts.append(text_input.strip())
        if srt_input:
            srt_text = self._extract_text_from_srt(srt_input)
            if srt_text:
                parts.append(srt_text)
        return " ".join(parts)
    
    def _strip_emotion_tags(self, text):
        """Strip Orpheus-specific emotion/expressive tags for non-Orpheus engines.
        
        Tags like <laugh>, <sigh>, <whisper>, etc. are only supported by Orpheus TTS.
        Other engines will fail if these tags are present in the text.
        """
        import re
        # Remove all <tag> style tags (Orpheus emotion/expressive tags)
        cleaned = re.sub(r'<[a-zA-Z_]+>', '', text)
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _synth_pyttsx3(self, text, output_file, voice_name, speed):
        """Synthesize using pyttsx3 (offline system voices)."""
        # Strip Orpheus-specific emotion tags (not supported by pyttsx3)
        text = self._strip_emotion_tags(text)
        
        # Call engine function
        engines.synth_pyttsx3(text, output_file, voice_name, speed)
    
    def _synth_edge_tts(self, text, output_file, voice_name, speed):
        """Synthesize using Microsoft Edge TTS via CLI (avoids async issues)."""
        # Strip Orpheus-specific emotion tags (not supported by edge-tts)
        text = self._strip_emotion_tags(text)
        
        # Call engine function
        engines.synth_edge_tts(text, output_file, voice_name, speed)
    
    def _synth_kokoro(self, text, output_file, voice, lang_code, speed, keep_models: bool = False):
        """Synthesize using Kokoro TTS via ONNX Runtime (lightweight 82M neural TTS).
        
        Uses kokoro-onnx package which works on Python 3.10-3.13.
        First run will download ~300MB model files.
        """
        # Strip Orpheus-specific emotion tags (not supported by Kokoro)
        text = self._strip_emotion_tags(text)
        
        # Call engine function
        engines.synth_kokoro(
            text, output_file, voice, lang_code, speed, keep_models,
            MODEL_MANAGER, tts_models_path, KOKORO_REPO_ID
        )
    
    def _synth_orpheus(self, text, output_file, lang, voice, keep_models: bool = False):
        """Synthesize using Orpheus TTS via llama.cpp (SOTA LLM-based TTS with emotions).
        
        Enhanced with multilingual support (24 voices across 8 languages) and long-form audio processing.
        Uses orpheus-cpp package which runs on CPU/GPU via llama.cpp backend.
        Works on Windows, Linux, and macOS without vLLM dependency.
        
        Features:
        - Supports inline emotion tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
        - Additional emotion descriptors for prompts (model-dependent): `happy`, `sad`, `angry`, `excited`, `surprised`, `whisper`, `fast`, `slow`, `crying`, etc.
        - Long-form processing: Sentence batching + crossfade stitching for better audio continuity
        
        Requirements:
        - Python 3.10-3.12 for pre-built CUDA wheels (Python 3.13 needs source build)
        - pip install orpheus-cpp llama-cpp-python
        
        Args:
            text: Input text to synthesize
            output_file: Path to save WAV output
            lang: Language (English, French, German, Korean, Hindi, Mandarin, Spanish, Italian)
            voice: Voice identifier (e.g., "en_tara", "fr_speaker_0")
        """
        # Call engine function
        engines.synth_orpheus(
            text, output_file, lang, voice, keep_models,
            MODEL_MANAGER, tts_orpheus_path, ORPHEUS_MODEL_ID, SNAC_REPO_ID
        )
    
    def _synth_csm(self, text, output_file, speaker_id, context_audio=None):
        """Synthesize using CSM (Conversational Speech Model) via HuggingFace Transformers.
        
        Premium conversational TTS (1 billion params) with speaker control.
        Uses native HuggingFace Transformers API (v4.52.1+).
        
        Default model: `unsloth/csm-1b` (Apache-2.0, public access).
        
        Emotional/Expressive Tags:
        CSM supports emotional control through text prompts:
        - Emotion: happy, sad, angry, empathetic, excited, calm, warm, cold, harsh, soft
        - Style: formal, casual, storytelling, dramatic, energetic, reassuring  
        - Expressiveness: whispering, shouting, enthusiastic, thoughtful, confident
        
        Examples:
        - "[0]I'm so excited to meet you!" - enthusiastic tone
        - "[1]I understand how you feel." - empathetic, warm tone
        - "[2]This is unacceptable!" - angry, harsh tone
        
        Args:
            text: Input text to synthesize
            output_file: Path to save WAV output  
            speaker_id: Speaker ID (0-9) for voice character
            context_audio: Optional previous audio for conversational continuity
        """
        # Call engine function
        engines.synth_csm(
            text, output_file, speaker_id, context_audio,
            MODEL_MANAGER, tts_csm_path, CSM_MODEL_ID, output_path
        )
    
    def _extract_text_from_srt(self, srt_path):
        """Extract plain text from SRT file."""
        if not srt_path or not os.path.exists(srt_path):
            return ""
        try:
            from srt import parse as SrtParse
            with open(srt_path, 'r', encoding='utf-8') as f:
                subtitles = list(SrtParse(f.read()))
            return " ".join(sub.content.strip() for sub in subtitles if sub.content.strip())
        except:
            return self._parse_srt_manual(srt_path)
    
    def _parse_srt_manual(self, srt_path):
        """Manually parse SRT file."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            texts = [l.strip() for l in lines if l.strip() and not l.strip().isdigit() and '-->' not in l]
            return " ".join(texts)
        except:
            return ""


# =============================================================================
# Node: TTSSLoadReferenceAudio
# =============================================================================
# NOTE: Reference audio loading node removed — XTTS-v2 voice cloning support has been removed.


# =============================================================================
# Node: TTSSLoadSRT
# =============================================================================
class TTSSLoadSRT:
    """Load an SRT subtitle file from the input directory."""
    
    @classmethod
    def INPUT_TYPES(cls):
        srt_files = ["(none)"]
        if os.path.exists(input_path):
            srt_files += sorted([
                f for f in os.listdir(input_path)
                if f.lower().endswith(('.srt', '.txt'))
            ])
        
        return {
            "required": {
                "srt_file": (srt_files, {"default": srt_files[0]}),
            }
        }
    
    RETURN_TYPES = ("SRT",)
    RETURN_NAMES = ("srt_path",)
    FUNCTION = "load_srt"
    CATEGORY = "🔊 TTSS"
    
    def load_srt(self, srt_file):
        if srt_file == "(none)":
            return (None,)
        srt_path = os.path.join(input_path, srt_file)
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"[TTSS] SRT file not found: {srt_path}")
        return (srt_path,)


# =============================================================================
# Node: TTSSLoadAudio
# =============================================================================
class TTSSLoadAudio:
    """Load an audio file from the input directory."""
    
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = ["(none)"]
        if os.path.exists(input_path):
            audio_files += sorted([
                f for f in os.listdir(input_path)
                if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))
            ])
        
        return {
            "required": {
                "audio": (audio_files, {"default": audio_files[0]}),
            }
        }
    
    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "load_audio"
    CATEGORY = "🔊 TTSS"
    
    def load_audio(self, audio):
        if audio == "(none)":
            return (None,)
        audio_path = os.path.join(input_path, audio)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"[TTSS] Audio file not found: {audio_path}")
        return (audio_path,)


# =============================================================================
# Node: TTSSPreviewAudio
# =============================================================================
class TTSSPreviewAudio:
    """Preview an audio file in the ComfyUI interface."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIOPATH",),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "🔊 TTSS"
    
    def preview(self, audio):
        if not audio or not os.path.exists(audio):
            return {"ui": {"audio": []}}
        
        audio_name = os.path.basename(audio)
        audio_dir = os.path.dirname(audio)
        
        if "output" in audio_dir.lower():
            audio_type = "output"
        elif "input" in audio_dir.lower():
            audio_type = "input"
        else:
            audio_type = "output"
        
        return {"ui": {"audio": [audio_name, audio_type]}}


# =============================================================================
# Node: TTSSCombineAudio
# =============================================================================
class TTSSCombineAudio:
    """Combine multiple audio files into one."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIOPATH",),
            },
            "optional": {
                "audio2": ("AUDIOPATH",),
                "audio3": ("AUDIOPATH",),
                "audio4": ("AUDIOPATH",),
                "gap_ms": ("INT", {"default": 500, "min": 0, "max": 5000, "step": 100}),
                "crossfade_ms": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 50}),
            }
        }
    
    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("combined_audio",)
    FUNCTION = "combine"
    CATEGORY = "🔊 TTSS"
    
    def combine(self, audio1, audio2=None, audio3=None, audio4=None, gap_ms=500, crossfade_ms=0):
        """Combine audio files with gaps or crossfade."""
        from pydub import AudioSegment
        
        combined = AudioSegment.from_file(audio1)
        
        for audio_path in [audio2, audio3, audio4]:
            if audio_path and os.path.exists(audio_path):
                next_audio = AudioSegment.from_file(audio_path)
                if crossfade_ms > 0:
                    combined = combined.append(next_audio, crossfade=crossfade_ms)
                else:
                    combined += AudioSegment.silent(duration=gap_ms)
                    combined += next_audio
        
        timestamp = int(time.time())
        output_file = os.path.join(output_path, f"ttss_combined_{timestamp}.wav")
        combined.export(output_file, format="wav")
        
        print(f"[TTSS] Combined audio saved: {output_file}")
        return (output_file,)


# =============================================================================
# Node: TTSSSaveAudio
# =============================================================================
class TTSSSaveAudio:
    """Save audio to a specific location with custom filename."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIOPATH",),
                "filename": ("STRING", {"default": "output"}),
            },
            "optional": {
                "format": (["wav", "mp3", "flac", "ogg"], {"default": "wav"}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_audio"
    CATEGORY = "🔊 TTSS"
    
    def save_audio(self, audio, filename, format="wav", add_timestamp=True):
        """Save audio to output directory."""
        from pydub import AudioSegment
        
        if not audio or not os.path.exists(audio):
            raise FileNotFoundError(f"[TTSS] Audio file not found: {audio}")
        
        # Build filename
        if add_timestamp:
            timestamp = int(time.time())
            out_name = f"{filename}_{timestamp}.{format}"
        else:
            out_name = f"{filename}.{format}"
        
        out_path = os.path.join(output_path, out_name)
        
        # Convert and save
        audio_seg = AudioSegment.from_file(audio)
        audio_seg.export(out_path, format=format)
        
        print(f"[TTSS] Audio saved: {out_path}")
        return (out_path,)


# =============================================================================
# Node: TTSConversation (Multi-speaker CSM conversations)
# =============================================================================
class TTSConversation:
    """
    Multi-speaker conversational TTS using CSM (Conversational Speech Model).
    Supports natural conversations with multiple speakers, emotional tags, and context awareness.
    
    Features:
    - Multiple speakers (0-9) in conversation format
    - Emotional/expressive tags for each speaker
    - Conversational context and continuity
    - Speaker-specific voice characteristics
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conversation_text": ("STRING", {
                    "default": "[0] Hello, how are you?\n[1] I'm doing well, thank you!\n[0] That's great to hear.",
                    "multiline": True
                }),
                "speaker_0_voice": ("STRING", {
                    "default": "0",
                    "tooltip": "CSM speaker ID for [0] (0-9)"
                }),
                "speaker_1_voice": ("STRING", {
                    "default": "1", 
                    "tooltip": "CSM speaker ID for [1] (0-9)"
                }),
                "speaker_2_voice": ("STRING", {
                    "default": "2",
                    "tooltip": "CSM speaker ID for [2] (0-9)"
                }),
                "speaker_3_voice": ("STRING", {
                    "default": "3",
                    "tooltip": "CSM speaker ID for [3] (0-9)"
                }),
            },
            "optional": {
                "context_audio": ("AUDIOPATH", {
                    "tooltip": "Previous conversation audio for continuity"
                }),
                "max_speakers": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Maximum number of speakers to process (1-10)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("conversation_audio",)
    FUNCTION = "generate_conversation"
    CATEGORY = "🔊 TTSS"

    def generate_conversation(self, conversation_text, speaker_0_voice="0", speaker_1_voice="1", 
                            speaker_2_voice="2", speaker_3_voice="3", context_audio=None, max_speakers=2):
        """Generate multi-speaker conversation audio using CSM."""
        
        # Validate input
        if not conversation_text or not conversation_text.strip():
            raise ValueError("[TTSS] No conversation text provided")
        
        # Validator warns if inline Orpheus tags are mixed into a CSM conversation
        try:
            validate_orpheus_tag_usage('csm', conversation_text)
        except Exception:
            pass
        
        # Map speaker IDs to voice IDs
        speaker_voice_map = {
            0: speaker_0_voice,
            1: speaker_1_voice, 
            2: speaker_2_voice,
            3: speaker_3_voice,
        }
        
        # Generate unique filename
        import hashlib
        text_hash = hashlib.md5(conversation_text.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_file = os.path.join(output_path, f"ttss_conversation_{timestamp}_{text_hash}.wav")
        
        # Call CSM engine function
        engines.synth_csm_conversation(
            conversation_text, output_file, speaker_voice_map,
            context_audio, max_speakers,
            MODEL_MANAGER, tts_csm_path, CSM_MODEL_ID, output_path
        )
        
        print(f"[TTSS] Conversation generated: {output_file}")
        return (output_file,)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "TTSSTextToSpeech",
    "TTSSLoadSRT",
    "TTSSLoadAudio",
    "TTSSPreviewAudio",
    "TTSSCombineAudio",
    "TTSSSaveAudio",
]
