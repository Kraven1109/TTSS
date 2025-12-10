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
SNAC_REPO_ID = "onnx-community/snac_24khz-ONNX"

# Orpheus model repositories by language
ORPHEUS_MODELS = {
    "English": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
    "Spanish": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
    "French": "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
    "German": "freddyaboulton/3b-de-ft-research_release-Q4_K_M-GGUF",
    "Italian": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
    "Hindi": "freddyaboulton/3b-hi-ft-research_release-Q4_K_M-GGUF",
    "Mandarin": "freddyaboulton/3b-zh-ft-research_release-Q4_K_M-GGUF",
    "Korean": "freddyaboulton/3b-ko-ft-research_release-Q4_K_M-GGUF",
}

def _get_edge_tts_cli():
    """Find edge-tts CLI executable path using system path."""
    # Check system PATH first
    if shutil.which("edge-tts"):
        return "edge-tts"
        
    # Fallback: Check Scripts folder in current Python environment
    import sys
    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")
    
    # Check for .exe (Windows) or no extension (Linux/Mac)
    for ext in [".exe", ""]:
        edge_path = os.path.join(scripts_dir, "edge-tts" + ext)
        if os.path.exists(edge_path):
            return edge_path
            
    # Final fallback
    return "edge-tts"


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

    def get_orpheus(self, lang: str = "English", n_gpu_layers: int = -1):
        key = f"orpheus:{lang}:{n_gpu_layers}"
        with self._lock(key):
            if key not in self._cache:
                try:
                    from orpheus_cpp import OrpheusCpp
                except Exception as e:
                    raise
                self._cache[key] = OrpheusCpp(verbose=False, lang=lang[:2].lower(), n_gpu_layers=n_gpu_layers)
            return self._cache[key]

    def unload_orpheus(self, lang: str = "English", n_gpu_layers: int = -1):
        key = f"orpheus:{lang}:{n_gpu_layers}"
        with self._lock(key):
            if key in self._cache:
                try:
                    del self._cache[key]
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


class TTSSSettings:
    """Node to control global TTSS settings like keeping models loaded across runs."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "keep_models": ("BOOL", {"default": False, "tooltip": "Keep models in memory after synthesis for faster subsequent runs (uses more RAM/GPU)"}),
                "csm_minimal_download": ("BOOL", {"default": False, "tooltip": "When True, download only a minimal subset of files for the CSM model to save disk space (may miss optional helper files)"}),
                "csm_cleanup_after_download": ("BOOL", {"default": False, "tooltip": "When True, delete non-essential files (e.g., .cache) from the local CSM model folder after download to free disk space"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "set_settings"
    CATEGORY = "🔊 TTSS"

    def set_settings(self, keep_models=False, csm_minimal_download=False, csm_cleanup_after_download=False):
        MODEL_MANAGER.keep_models_default = bool(keep_models)
        MODEL_MANAGER.csm_minimal_download_default = bool(csm_minimal_download)
        MODEL_MANAGER.csm_cleanup_after_download_default = bool(csm_cleanup_after_download)
        # Note: Keep the defaults off to avoid missing optional files unless user opts in
        status = (f"Keep models: {MODEL_MANAGER.keep_models_default}; CSM minimal download: {MODEL_MANAGER.csm_minimal_download_default}; "
                 f"CSM cleanup: {MODEL_MANAGER.csm_cleanup_after_download_default}")
        print(f"[TTSS] {status}")
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
                "show_orpheus_help": ("BOOL", {"default": False, "tooltip": "Show Orpheus inline tags & descriptors in logs and warnings"}),
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
                   text_input=None, srt_input=None, context_audio=None, show_orpheus_help=False):
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
                self._synth_kokoro(final_text, output_file, voice_name, kokoro_lang, speed, keep_models=False)
            elif engine == "orpheus":
                self._synth_orpheus(final_text, output_file, orpheus_lang, voice_name, keep_models=False)
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
        import pyttsx3
        
        # Strip Orpheus-specific emotion tags (not supported by pyttsx3)
        text = self._strip_emotion_tags(text)
        
        engine = pyttsx3.init()
        
        # Set speed
        default_rate = engine.getProperty('rate')
        engine.setProperty('rate', int(default_rate * speed))
        
        # Set voice
        if voice_name:
            voices = engine.getProperty('voices')
            for v in voices:
                if voice_name.lower() in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break
        
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        engine.stop()
    
    def _synth_edge_tts(self, text, output_file, voice_name, speed):
        """Synthesize using Microsoft Edge TTS via CLI (avoids async issues)."""
        voice = voice_name if voice_name else "en-US-AriaNeural"
        
        # Strip Orpheus-specific emotion tags (not supported by edge-tts)
        text = self._strip_emotion_tags(text)
        
        # Use edge-tts CLI to avoid asyncio conflicts with ComfyUI
        edge_tts_cmd = _get_edge_tts_cli()
        cmd = [
            edge_tts_cmd,
            '--voice', voice,
            '--text', text,
            '--write-media', output_file
        ]
        
        # Only add --rate if speed is not 1.0
        if speed != 1.0:
            rate_percent = int((speed - 1) * 100)
            rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
            cmd.extend(['--rate', rate])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"edge-tts failed: {result.stderr}")
        except FileNotFoundError:
            # edge-tts CLI not in PATH, try Python API with thread
            self._synth_edge_tts_threaded(text, output_file, voice, speed)
    
    def _synth_edge_tts_threaded(self, text, output_file, voice, speed):
        """Fallback: Run edge-tts in a separate thread to avoid async conflicts."""
        import asyncio
        import edge_tts
        
        # Calculate rate string for edge_tts API
        if speed != 1.0:
            rate_percent = int((speed - 1) * 100)
            rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
        else:
            rate = "+0%"
        
        result_holder = {'error': None}
        
        def run_in_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def synth():
                    communicate = edge_tts.Communicate(text, voice, rate=rate)
                    await communicate.save(output_file)
                
                loop.run_until_complete(synth())
                loop.close()
            except Exception as e:
                result_holder['error'] = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=300)  # 5 minute timeout
        
        if result_holder['error']:
            raise result_holder['error']
    
    def _synth_kokoro(self, text, output_file, voice, lang_code, speed, keep_models: bool = False):
        """Synthesize using Kokoro TTS via ONNX Runtime (lightweight 82M neural TTS).
        
        Uses kokoro-onnx package which works on Python 3.10-3.13.
        First run will download ~300MB model files.
        """
        # Strip Orpheus-specific emotion tags (not supported by Kokoro)
        text = self._strip_emotion_tags(text)
        
        try:
            import numpy as np
            from kokoro_onnx import Kokoro
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "[TTSS] kokoro-onnx not installed. Run: pip install kokoro-onnx soundfile\n"
                "Note: First run will download ~300MB model (or ~80MB quantized)."
            )
        
        # Model paths in TTS models directory
        kokoro_models_path = os.path.join(tts_models_path, "kokoro")
        os.makedirs(kokoro_models_path, exist_ok=True)
        model_path = os.path.join(kokoro_models_path, "kokoro-v1.0.onnx")
        voices_path = os.path.join(kokoro_models_path, "voices-v1.0.bin")
        
        # Auto-download models if not present
        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            print("[TTSS] Downloading Kokoro ONNX models (~300MB)...")
            self._download_kokoro_models(model_path, voices_path)
        
        # Map lang_code to kokoro-onnx lang format
        lang_map = {
            "a": "en-us",  # American English
            "b": "en-gb",  # British English
            "e": "es",     # Spanish
            "f": "fr-fr",  # French
            "h": "hi",     # Hindi
            "i": "it",     # Italian
            "j": "ja",     # Japanese
            "p": "pt-br",  # Brazilian Portuguese
            "z": "zh",     # Mandarin Chinese
        }
        lang = lang_map.get(lang_code, "en-us")
        
        # Initialize Kokoro with ONNX model (possibly cached)
        kokoro = MODEL_MANAGER.get_kokoro(model_path, voices_path)
        
        # Long-form audio processing for Kokoro (phoneme limit ~510)
        import re
        
        # Split text into sentences (handle common sentence endings)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Estimate phoneme count (rough approximation: ~1.5 phonemes per character for English)
        # Kokoro has a hard limit of ~510 phonemes
        max_phonemes_per_chunk = 400  # Conservative limit to avoid truncation
        
        # If text is short, process as single chunk
        if len(sentences) <= 1 or len(text) < 200:
            text_chunks = [text]
        else:
            # Group sentences into chunks based on estimated phoneme count
            text_chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Rough phoneme estimation
                estimated_phonemes = len(current_chunk + sentence) * 1.5
                
                if estimated_phonemes < max_phonemes_per_chunk:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                text_chunks.append(current_chunk.strip())
        
        print(f"[TTSS] Processing {len(text_chunks)} text chunks for Kokoro")
        
        # Process each chunk and collect audio
        all_audio_chunks = []
        sample_rate = None
        
        # Helper to process Kokoro text safely by recursively splitting when phoneme limits are hit
        def _process_kokoro_chunk(text_to_process):
            try:
                return kokoro.create(text_to_process, voice=voice, speed=speed, lang=lang)
            except Exception as ke:
                em = str(ke).lower()
                if ("index" in em and "out of bounds" in em) or ("phoneme" in em) or ("truncat" in em):
                    words = text_to_process.split()
                    if len(words) <= 3:
                        raise
                    mid = len(words) // 2
                    left = " ".join(words[:mid]).strip()
                    right = " ".join(words[mid:]).strip()
                    left_res = _process_kokoro_chunk(left)
                    right_res = _process_kokoro_chunk(right)
                    # left_res and right_res are tuples (samples, sr)
                    left_samps, left_sr = left_res
                    right_samps, right_sr = right_res
                    if left_sr != right_sr:
                        # Resample right_samps to left_sr if needed (optional, torchaudio)
                        try:
                            import torchaudio
                            import torch
                            resampler = torchaudio.transforms.Resample(right_sr, left_sr)
                            right_samps = resampler(torch.tensor(right_samps)).numpy()
                            right_sr = left_sr
                        except Exception:
                            pass
                    return (np.concatenate([left_samps, right_samps], axis=0), left_sr)
                else:
                    raise

        for i, chunk_text in enumerate(text_chunks):
            print(f"[TTSS] Processing Kokoro chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
            
            try:
                # Generate speech for this chunk using safe helper
                samples, sr = _process_kokoro_chunk(chunk_text)
                
                if sample_rate is None:
                    sample_rate = sr
                
                # Convert to numpy array and ensure samples is 1D
                samples = np.asarray(samples)
                if samples.ndim > 1:
                    samples = samples.flatten()
                
                all_audio_chunks.append(samples)
                
            except Exception as e:
                error_msg = str(e)
                if "index" in error_msg.lower() and "out of bounds" in error_msg.lower():
                    print(f"[TTSS] Chunk too long, splitting further: {chunk_text[:30]}...")
                    # If still too long, split this chunk into smaller pieces
                    words = chunk_text.split()
                    sub_chunks = []
                    current_sub = ""
                    
                    for word in words:
                        if len(current_sub + word) * 1.5 < max_phonemes_per_chunk / 2:
                            current_sub += word + " "
                        else:
                            if current_sub:
                                sub_chunks.append(current_sub.strip())
                            current_sub = word + " "
                    
                    if current_sub:
                        sub_chunks.append(current_sub.strip())
                    
                    # Process sub-chunks
                    for sub_chunk in sub_chunks:
                        try:
                            sub_samples, sub_sr = kokoro.create(sub_chunk, voice=voice, speed=speed, lang=lang)
                            # Ensure numpy array and flatten
                            sub_samples = np.asarray(sub_samples)
                            if sub_samples.ndim > 1:
                                sub_samples = sub_samples.flatten()
                            all_audio_chunks.append(sub_samples)
                        except Exception as sub_e:
                            print(f"[TTSS] Skipping problematic sub-chunk: {sub_e}")
                            continue
                else:
                    print(f"[TTSS] Skipping problematic chunk: {e}")
                    continue
        
        # Concatenate all audio chunks
        if all_audio_chunks:
            # Simple concatenation (Kokoro outputs are already properly formatted)
            final_audio = np.concatenate(all_audio_chunks, axis=0)
        else:
            raise RuntimeError("[TTSS] Kokoro generated no audio")
        
        # Save audio
        sf.write(output_file, final_audio, sample_rate)
        # Decide final keep_models policy (per-node overrides global)
        effective_keep = bool(keep_models) or getattr(MODEL_MANAGER, 'keep_models_default', False)
        # Unload Kokoro if not expected to keep the model
        if not effective_keep:
            try:
                MODEL_MANAGER.unload_kokoro(model_path, voices_path)
            except Exception:
                pass
    
    def _download_kokoro_models(self, model_path, voices_path):
        """Download Kokoro ONNX model files using HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "[TTSS] huggingface_hub not installed. Run: pip install huggingface-hub"
            )
        
        repo_id = KOKORO_REPO_ID
        
        # Download model file
        if not os.path.exists(model_path):
            print(f"[TTSS] Downloading kokoro-v1.0.onnx...")
            downloaded_model = hf_hub_download(
                repo_id=repo_id,
                filename="onnx/kokoro-v1.0.onnx",
                local_dir=os.path.dirname(model_path),
                local_dir_use_symlinks=False,
            )
            # Move to expected location
            os.rename(downloaded_model, model_path)
        
        # Download voices file
        if not os.path.exists(voices_path):
            print(f"[TTSS] Downloading voices-v1.0.bin...")
            downloaded_voices = hf_hub_download(
                repo_id=repo_id,
                filename="voices/voices-v1.0.bin",
                local_dir=os.path.dirname(voices_path),
                local_dir_use_symlinks=False,
            )
            # Move to expected location
            os.rename(downloaded_voices, voices_path)
        
        print("[TTSS] Kokoro models downloaded successfully!")
    
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
        import numpy as np
        from scipy.io.wavfile import write as wav_write
        
        # Check for llama-cpp-python first (more likely to fail)
        try:
            import llama_cpp
        except ImportError:
            raise ImportError(
                "[TTSS] llama-cpp-python not installed or incompatible.\n\n"
                "For Python 3.10-3.12 with CUDA:\n"
                "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n\n"
                "For Python 3.10-3.12 CPU only:\n"
                "  pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n\n"
                "⚠️ Python 3.13: No pre-built wheels available. Requires building from source with CUDA toolkit.\n"
                "   See: https://github.com/abetlen/llama-cpp-python#installation"
            )
        
        try:
            from orpheus_cpp import OrpheusCpp
        except ImportError:
            raise ImportError(
                "[TTSS] Orpheus TTS not installed. Run:\n"
                "  pip install orpheus-cpp\n\n"
                "Note: First run will download ~3GB GGUF model."
            )
        
        
        orpheus_repo = ORPHEUS_MODELS.get(lang, ORPHEUS_MODELS["English"])  # Default to English
        snac_repo = SNAC_REPO_ID
        
        # Download main Orpheus model
        orpheus_model_path = os.path.join(tts_orpheus_path, os.path.basename(orpheus_repo))
        if not os.path.exists(orpheus_model_path):
            print(f"[TTSS] Downloading Orpheus model to: {orpheus_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=orpheus_repo,
                local_dir=orpheus_model_path,
                local_dir_use_symlinks=False,
            )
        
        # Download SNAC model
        snac_model_path = os.path.join(tts_orpheus_path, "snac_24khz-ONNX")
        if not os.path.exists(snac_model_path):
            print(f"[TTSS] Downloading SNAC model to: {snac_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=snac_repo,
                local_dir=snac_model_path,
                local_dir_use_symlinks=False,
            )
        
        # Initialize Orpheus with llama.cpp backend (GPU enabled)
        # Long-form audio processing: Split text into sentences for better quality
        import re
    
        # Split text into sentences (handle common sentence endings)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
    
        # If text is short, process as single chunk
        if len(sentences) <= 1 or len(text) < 500:
            text_chunks = [text]
        else:
            # Group sentences into chunks (aim for ~200-300 words per chunk)
            text_chunks = []
            current_chunk = ""
        
            for sentence in sentences:
                if len(current_chunk + sentence) < 300:  # Character count approximation
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        text_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
        
            if current_chunk:
                text_chunks.append(current_chunk.strip())
    
        print(f"[TTSS] Processing {len(text_chunks)} text chunks for long-form audio")
    
        # Initialize Orpheus with correct language
        lang_code = lang[:2].lower()  # "en", "fr", "de", etc.
        # Use cached orpheus instance if available (ModelManager)
        n_gpu_layers = -1
        orpheus = MODEL_MANAGER.get_orpheus(lang, n_gpu_layers=n_gpu_layers)
    
        # Process each chunk and collect audio
        all_audio_chunks = []
        sample_rate = None
    
        for i, chunk_text in enumerate(text_chunks):
            print(f"[TTSS] Processing chunk {i+1}/{len(text_chunks)}: {chunk_text[:50]}...")
        
            # Generate speech for this chunk
            chunk_buffer = []
            chunk_sample_rate = None
        
            for j, (sr, audio_chunk) in enumerate(orpheus.stream_tts_sync(chunk_text, options={"voice_id": voice})):
                    if chunk_sample_rate is None:
                        chunk_sample_rate = sr
                        if sample_rate is None:
                            sample_rate = sr
                    # Normalize and convert to mono/float32 for safe concatenation
                    def _to_mono_float32(arr):
                        arr = np.asarray(arr)
                        # Convert integer arrays to float32 normalized -1..1
                        if arr.dtype.kind in ('i', 'u'):
                            try:
                                maxv = np.iinfo(arr.dtype).max
                                arr = arr.astype(np.float32) / float(maxv)
                            except Exception:
                                arr = arr.astype(np.float32)
                        else:
                            arr = arr.astype(np.float32)
                        if arr.ndim > 1:
                            # Many audio libraries return shape (channels, samples) or (samples, channels)
                            if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
                                # shape (channels, samples)
                                arr = arr.mean(axis=0)
                            else:
                                # shape (samples, channels)
                                arr = arr.mean(axis=1)
                        return arr
                    audio_chunk = _to_mono_float32(audio_chunk)
                    chunk_buffer.append(audio_chunk)
        
            if chunk_buffer:
                # Concatenate chunk audio (ensure numpy arrays)
                chunk_audio = np.concatenate([np.asarray(x, dtype=np.float32) for x in chunk_buffer], axis=0)
                all_audio_chunks.append(chunk_audio)
    
        # If we failed to capture a sample rate, fall back to 24000
        if sample_rate is None and len(all_audio_chunks) > 0:
            sample_rate = 24000

        # Crossfade stitching between chunks
        if len(all_audio_chunks) > 1:
            print("[TTSS] Applying crossfade stitching between audio chunks")

            # Crossfade parameters (200ms at 24kHz)
            crossfade_samples = int(0.2 * sample_rate)  # 200ms crossfade

            stitched_audio = [all_audio_chunks[0]]  # First chunk unchanged

            for i in range(1, len(all_audio_chunks)):
                prev_chunk = stitched_audio[-1]
                curr_chunk = all_audio_chunks[i]

                # Ensure we have enough samples for crossfade
                crossfade_len = min(crossfade_samples, len(prev_chunk), len(curr_chunk))

                if crossfade_len > 0:
                    # Create crossfade window (linear fade out/in)
                    fade_out = np.linspace(1.0, 0.0, crossfade_len)
                    fade_in = np.linspace(0.0, 1.0, crossfade_len)

                    # Apply crossfade
                    prev_end = prev_chunk[-crossfade_len:]
                    curr_start = curr_chunk[:crossfade_len]

                    # Mix the overlapping regions
                    mixed_region = prev_end * fade_out + curr_start * fade_in

                    # Combine: prev_chunk (without overlap) + mixed_region + curr_chunk (without overlap)
                    combined = np.concatenate([
                        prev_chunk[:-crossfade_len],
                        mixed_region,
                        curr_chunk[crossfade_len:]
                    ])

                    stitched_audio[-1] = combined
                else:
                    # No crossfade possible, just concatenate
                    stitched_audio[-1] = np.concatenate([prev_chunk, curr_chunk])

            # Final concatenation
            audio = stitched_audio[0]
        else:
            # Single chunk, no stitching needed
            audio = all_audio_chunks[0] if all_audio_chunks else np.array([])
    
        # Save final audio (convert to int16 for wider player compatibility)
        if len(audio) > 0:
            # Ensure numpy float32 in -1..1 before scaling to int16
            audio = np.asarray(audio, dtype=np.float32)
            max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
            if max_abs > 1.0:
                # Normalize by maximum to avoid clipping
                audio = audio / max_abs
            # Convert to int16
            int_audio = (audio * 32767.0).astype(np.int16)
            wav_write(output_file, sample_rate, int_audio)
        # Decide per-node keep vs global default
        effective_keep = bool(keep_models) or getattr(MODEL_MANAGER, 'keep_models_default', False)
        # Unload the orpheus instance if configured to not keep models loaded
        if not effective_keep:
            try:
                MODEL_MANAGER.unload_orpheus(lang, n_gpu_layers=n_gpu_layers)
            except Exception:
                pass
        # If no audio generated, raise
        if len(audio) == 0:
            raise RuntimeError("[TTSS] Orpheus generated no audio")
    
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
        tmp_context_path = None # Initialize variable
        try:
            import torch
            import torchaudio
            from transformers import CsmForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"[TTSS] CSM requires: pip install transformers>=4.52.1 torchaudio accelerate\n"
                f"Missing: {e}"
            )
        
        try:
            # Check for CUDA (CSM is heavy, GPU strongly recommended)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("[TTSS] WARNING: CSM on CPU will be very slow. GPU strongly recommended!")
            
            # Model ID: unsloth/csm-1b (community downstream of sesame's CSM)
            # unsloth/csm-1b is Apache-2.0 and does not require gate access like sesame's gated model did.
            model_id = CSM_MODEL_ID
            
            print(f"[TTSS] Loading CSM model from {model_id}...")
            
            # Download model to ComfyUI directory (not user cache)
            local_model_path = os.path.join(tts_csm_path, model_id.replace("/", "_"))
            if not os.path.exists(local_model_path):
                print(f"[TTSS] Downloading CSM model to: {local_model_path}")
                from huggingface_hub import snapshot_download
                try:
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=local_model_path,
                        local_dir_use_symlinks=False,
                    )
                    print(f"[TTSS] CSM model downloaded successfully")
                except Exception as e:
                    raise RuntimeError(f"[TTSS] Failed to download CSM model: {e}") from e
            
            # Use cached model via ModelManager
            try:
                processor, model = MODEL_MANAGER.get_csm(local_model_path, device)
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
                    raise RuntimeError(
                        f"[TTSS] CSM model access denied: {e}.\n"
                        f"If you think this is a permissions issue, please check your HuggingFace account or token and ensure the model repo is available (unsloth/csm-1b).\n"
                        f"Original error: {e}"
                    )
                raise
            
            # Build conversation with speaker ID
            # CSM expects role=speaker_id (as string) and content as list of typed dicts
            conversation = [
                {
                    "role": f"{speaker_id}",
                    "content": [{"type": "text", "text": text}]
                }
            ]
            
            # If context audio provided, add it as conversation history
            if context_audio and os.path.exists(context_audio):
                print(f"[TTSS] Using context audio: {context_audio}")
                context_waveform, context_sr = torchaudio.load(context_audio)
                # Resample to 24kHz if needed (CSM uses 24kHz)
                audio_to_pass = context_audio
                if context_sr != 24000:
                    resampler = torchaudio.transforms.Resample(context_sr, 24000)
                    context_waveform = resampler(context_waveform)
                    # Create a safe temp file that we can track
                    fd, tmp_context_path = tempfile.mkstemp(suffix=".wav", dir=output_path)
                    os.close(fd) # Close file descriptor immediately so torchaudio can write to it
                    torchaudio.save(tmp_context_path, context_waveform, 24000)
                    audio_to_pass = tmp_context_path
                # Add context as previous turn (audio comes from a file path)
                conversation.insert(0, {
                    "role": f"{speaker_id}", 
                    "content": [
                        {"type": "text", "text": ""},  # Empty text for context turn
                        {"type": "audio", "path": audio_to_pass}
                    ]
                })
            
            # Process inputs
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            ).to(device)
            
            # Generate audio
            print(f"[TTSS] Generating speech with CSM (speaker={speaker_id})...")
            with torch.no_grad():
                audio_output = model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=2048,  # ~85 seconds at 24kHz
                )
            
            # Save audio (CSM outputs 24kHz)
            processor.save_audio(audio_output, output_file)
            
            print(f"[TTSS] CSM synthesis complete: {output_file}")

        finally:
            # [FIX] Clean up the temp file even if errors occur
            if tmp_context_path and os.path.exists(tmp_context_path):
                try:
                    os.remove(tmp_context_path)
                except Exception:
                    print(f"[TTSS] Warning: Could not remove temp file {tmp_context_path}")

            # Always unload CSM after use
            effective_keep = False
            # Unload CSM
            if not effective_keep:
                try:
                    MODEL_MANAGER.unload_csm(local_model_path, device)
                except Exception:
                    pass
    
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
        
        # Parse conversation text
        lines = [line.strip() for line in conversation_text.split('\n') if line.strip()]
        conversation_turns = []
        
        for line in lines:
            # Parse speaker tags like [0], [1], etc.
            import re
            match = re.match(r'^\[(\d+)\]\s*(.+)$', line)
            if match:
                speaker_id = int(match.group(1))
                text = match.group(2).strip()
                if speaker_id < max_speakers:  # Only process speakers within max_speakers
                    # Validator warns if inline Orpheus tags are mixed into a CSM conversation
                    try:
                        validate_orpheus_tag_usage('csm', text)
                    except Exception:
                        pass
                    conversation_turns.append({
                        'speaker': speaker_id,
                        'text': text
                    })
        
        if not conversation_turns:
            raise ValueError("[TTSS] No valid conversation turns found. Use format: [0] Speaker text")
        
        print(f"[TTSS] Processing {len(conversation_turns)} conversation turns")
        
        # Map speaker IDs to voice IDs
        speaker_voice_map = {
            0: speaker_0_voice,
            1: speaker_1_voice, 
            2: speaker_2_voice,
            3: speaker_3_voice,
        }
        
        # Build conversation for CSM
        conversation = []
        
        # Add context audio if provided
        if context_audio and os.path.exists(context_audio):
            print(f"[TTSS] Using context audio: {context_audio}")
            import torchaudio
            context_waveform, context_sr = torchaudio.load(context_audio)
            # Resample to 24kHz if needed
            audio_to_pass = context_audio
            if context_sr != 24000:
                import torch
                resampler = torchaudio.transforms.Resample(context_sr, 24000)
                context_waveform = resampler(context_waveform)
                # Save resampled audio to temporary file
                tmp_context_path = os.path.join(output_path, f"ttss_context_{int(time.time())}.wav")
                torchaudio.save(tmp_context_path, context_waveform, 24000)
                audio_to_pass = tmp_context_path
            # Add as previous turn
            conversation.append({
                "role": "0",  # Default context speaker
                "content": [
                    {"type": "text", "text": ""},
                    {"type": "audio", "path": audio_to_pass}
                ]
            })
        
        # Add conversation turns
        for turn in conversation_turns:
            speaker_id = turn['speaker']
            voice_id = speaker_voice_map.get(speaker_id, str(speaker_id))
            
            conversation.append({
                "role": voice_id,
                "content": [{"type": "text", "text": turn['text']}]
            })
        
        # Generate unique filename
        import hashlib
        import time
        text_hash = hashlib.md5(conversation_text.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        output_file = os.path.join(output_path, f"ttss_conversation_{timestamp}_{text_hash}.wav")
        
        # Use CSM synthesis
        self._synth_csm_conversation(conversation, output_file)
        
        print(f"[TTSS] Conversation generated: {output_file}")
        return (output_file,)
    
    def _synth_csm_conversation(self, conversation, output_file):
        """Synthesize conversation using CSM with multiple speakers."""
        
        try:
            import torch
            import torchaudio
            from transformers import CsmForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"[TTSS] CSM requires: pip install transformers>=4.52.1 torchaudio accelerate\n"
                f"Missing: {e}"
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[TTSS] WARNING: CSM on CPU will be very slow. GPU strongly recommended!")
        
        model_id = CSM_MODEL_ID
        print(f"[TTSS] Loading CSM model for conversation...")
        
        # Download model to ComfyUI directory
        local_model_path = os.path.join(tts_csm_path, model_id.replace("/", "_"))
        if not os.path.exists(local_model_path):
            print(f"[TTSS] Downloading CSM model to: {local_model_path}")
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False,
                )
                print(f"[TTSS] CSM model downloaded successfully")
            except Exception as e:
                raise RuntimeError(f"[TTSS] Failed to download CSM model: {e}") from e
        
        # Use cached model from ModelManager (may load once)
        try:
            processor, model = MODEL_MANAGER.get_csm(local_model_path, device)
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
                raise RuntimeError(
                    f"[TTSS] CSM model access denied: {e}.\n"
                    f"If you think this is a permissions issue, please check your HuggingFace account or token and ensure the model repo is available (unsloth/csm-1b).\n"
                    f"Original error: {e}"
                )
            raise
        
        # Process conversation
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(device)
        
        print(f"[TTSS] Generating conversation with {len(conversation)} turns...")
        with torch.no_grad():
            audio_output = model.generate(
                **inputs,
                output_audio=True,
                max_new_tokens=2048 * len(conversation),  # Scale with conversation length
            )
        
        # Save audio
        processor.save_audio(audio_output, output_file)
        print(f"[TTSS] Conversation synthesis complete: {output_file}")
        # Always unload CSM after use
        effective_keep = False
        # Unload model to free resources
        if not effective_keep:
            try:
                MODEL_MANAGER.unload_csm(local_model_path, device)
            except Exception:
                pass


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
