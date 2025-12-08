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

# Orpheus voices (built-in)
ORPHEUS_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

# Orpheus emotion tags
ORPHEUS_EMOTIONS = ["<laugh>", "<chuckle>", "<sigh>", "<cough>", "<sniffle>", "<groan>", "<yawn>", "<gasp>"]

# CSM voices (speaker IDs for conversational TTS)
CSM_VOICES = [str(i) for i in range(10)]  # 0-9 speaker IDs

def _get_edge_tts_cli():
    """Find edge-tts CLI executable path."""
    import sys
    # Try Scripts folder in Python environment
    python_dir = os.path.dirname(sys.executable)
    scripts_dir = os.path.join(python_dir, "Scripts")
    edge_tts_exe = os.path.join(scripts_dir, "edge-tts.exe")
    if os.path.exists(edge_tts_exe):
        return edge_tts_exe
    # Try without .exe (Linux/Mac)
    edge_tts_bin = os.path.join(scripts_dir, "edge-tts")
    if os.path.exists(edge_tts_bin):
        return edge_tts_bin
    # Fallback to PATH
    return "edge-tts"

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
    """Get available Orpheus voices."""
    return ORPHEUS_VOICES

def get_csm_voices():
    """Get available CSM speaker IDs."""
    return CSM_VOICES

def get_reference_audio_files():
    """Get reference audio files for voice cloning."""
    files = ["(none)"]
    if os.path.exists(tts_reference_path):
        for f in sorted(os.listdir(tts_reference_path)):
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                files.append(f)
    return files


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
                "orpheus_voice": (orpheus_voices, {"default": "tara"}),
                "csm_voice": (csm_voices, {"default": "0"}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True, "multiline": True}),
                "srt_input": ("SRT",),
                "reference_audio": ("AUDIOPATH",),
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
                   orpheus_voice="tara", csm_voice="0", reference_audio="(none)",
                   text_input=None, srt_input=None, context_audio=None):
        """Generate speech from text using selected engine."""
        
        # Build final text
        final_text = self._build_text(text, text_input, srt_input)
        if not final_text:
            raise ValueError("[TTSS] No text provided for synthesis")
        
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
        
        # Handle reference audio for XTTS voice cloning
        ref_audio_path = None
        if reference_audio and reference_audio != "(none)":
            if os.path.isabs(reference_audio):
                ref_audio_path = reference_audio
            else:
                ref_audio_path = os.path.join(tts_reference_path, reference_audio)
        
        # Route to appropriate engine
        if engine == "pyttsx3":
            self._synth_pyttsx3(final_text, output_file, voice_name, speed)
        elif engine == "edge-tts":
            self._synth_edge_tts(final_text, output_file, voice_name, speed)
        elif engine == "kokoro":
            self._synth_kokoro(final_text, output_file, voice_name, kokoro_lang, speed)
        elif engine == "orpheus":
            self._synth_orpheus(final_text, output_file, voice_name)
        elif engine == "csm":
            self._synth_csm(final_text, output_file, voice_name, context_audio)
        else:
            raise ValueError(f"[TTSS] Unknown engine: {engine}")
        
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
    
    def _synth_kokoro(self, text, output_file, voice, lang_code, speed):
        """Synthesize using Kokoro TTS via ONNX Runtime (lightweight 82M neural TTS).
        
        Uses kokoro-onnx package which works on Python 3.10-3.13.
        First run will download ~300MB model files.
        """
        # Strip Orpheus-specific emotion tags (not supported by Kokoro)
        text = self._strip_emotion_tags(text)
        
        try:
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
        
        # Initialize Kokoro with ONNX model
        kokoro = Kokoro(model_path, voices_path)
        
        # Generate speech
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        
        # Save audio
        sf.write(output_file, samples, sample_rate)
    
    def _download_kokoro_models(self, model_path, voices_path):
        """Download Kokoro ONNX model files."""
        import urllib.request
        
        base_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
        
        # Download model file
        if not os.path.exists(model_path):
            print(f"[TTSS] Downloading kokoro-v1.0.onnx...")
            urllib.request.urlretrieve(f"{base_url}/kokoro-v1.0.onnx", model_path)
        
        # Download voices file
        if not os.path.exists(voices_path):
            print(f"[TTSS] Downloading voices-v1.0.bin...")
            urllib.request.urlretrieve(f"{base_url}/voices-v1.0.bin", voices_path)
        
        print("[TTSS] Kokoro models downloaded successfully!")
    
    def _synth_orpheus(self, text, output_file, voice):
        """Synthesize using Orpheus TTS via llama.cpp (SOTA LLM-based TTS with emotions).
        
        Uses orpheus-cpp package which runs on CPU/GPU via llama.cpp backend.
        Works on Windows, Linux, and macOS without vLLM dependency.
        
        Supports emotion tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
        Additional expressive tags: <happy>, <normal>, <disgust>, <sad>, <frustrated>, <slow>, <excited>, 
        <whisper>, <panicky>, <curious>, <surprise>, <fast>, <crying>, <deep>, <sleepy>, <angry>, <high>, <shout>
        
        Requirements:
        - Python 3.10-3.12 for pre-built CUDA wheels (Python 3.13 needs source build)
        - pip install orpheus-cpp llama-cpp-python
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
        
        # Pre-download Orpheus models to ComfyUI directory (like ComfyUI_Qwen3-VL-Instruct)
        lang_to_model = {
            "en": "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF",
            "es": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
            "fr": "freddyaboulton/3b-fr-ft-research_release-Q4_K_M-GGUF",
            "de": "freddyaboulton/3b-de-ft-research_release-Q4_K_M-GGUF",
            "it": "freddyaboulton/3b-es_it-ft-research_release-Q4_K_M-GGUF",
            "hi": "freddyaboulton/3b-hi-ft-research_release-Q4_K_M-GGUF",
            "zh": "freddyaboulton/3b-zh-ft-research_release-Q4_K_M-GGUF",
            "ko": "freddyaboulton/3b-ko-ft-research_release-Q4_K_M-GGUF",
        }
        
        orpheus_repo = lang_to_model.get("en", lang_to_model["en"])  # Default to English
        snac_repo = "onnx-community/snac_24khz-ONNX"
        
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
        # Set HF environment variables to point to our downloaded models
        old_hf_home = os.environ.get("HF_HOME")
        old_hf_hub_cache = os.environ.get("HF_HUB_CACHE")
        old_huggingface_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
        
        os.environ["HF_HOME"] = tts_orpheus_path
        os.environ["HF_HUB_CACHE"] = tts_orpheus_path
        os.environ["HUGGINGFACE_HUB_CACHE"] = tts_orpheus_path
        
        try:
            orpheus = OrpheusCpp(verbose=False, lang="en", n_gpu_layers=-1)
            
            # Generate speech with streaming
            buffer = []
            sample_rate = None
            for i, (sr, chunk) in enumerate(orpheus.stream_tts_sync(text, options={"voice_id": voice})):
                if sample_rate is None:
                    sample_rate = sr
                # Ensure chunk is 1D for concatenation
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                buffer.append(chunk)
            
            if buffer:
                # Concatenate all chunks and save
                audio = np.concatenate(buffer, axis=0)  # Concatenate along time axis
                wav_write(output_file, sample_rate, audio)
            else:
                raise RuntimeError("[TTSS] Orpheus generated no audio")
        finally:
            # Restore original environment variables
            if old_hf_home is not None:
                os.environ["HF_HOME"] = old_hf_home
            else:
                os.environ.pop("HF_HOME", None)
                
            if old_hf_hub_cache is not None:
                os.environ["HF_HUB_CACHE"] = old_hf_hub_cache
            else:
                os.environ.pop("HF_HUB_CACHE", None)
                
            if old_huggingface_hub_cache is not None:
                os.environ["HUGGINGFACE_HUB_CACHE"] = old_huggingface_hub_cache
            else:
                os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    
    def _synth_csm(self, text, output_file, speaker_id, context_audio=None):
        """Synthesize using CSM (Conversational Speech Model) via HuggingFace Transformers.
        
        Premium conversational TTS (1B params) with speaker control.
        Uses native HuggingFace Transformers API (v4.52.1+).
        
        NOTE: sesame/csm-1b is a GATED model - requires HuggingFace login:
            huggingface-cli login
        
        Args:
            text: Input text to synthesize
            output_file: Path to save WAV output  
            speaker_id: Speaker ID (0-9) for voice character
            context_audio: Optional previous audio for conversational continuity
        """
        try:
            import torch
            import torchaudio
            from transformers import CsmForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"[TTSS] CSM requires: pip install transformers>=4.52.1 torchaudio accelerate\n"
                f"Missing: {e}"
            )
        
        # Check for CUDA (CSM is heavy, GPU strongly recommended)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("[TTSS] WARNING: CSM on CPU will be very slow. GPU strongly recommended!")
        
        # Model ID (gated - requires HuggingFace login)
        model_id = "sesame/csm-1b"
        
        print(f"[TTSS] Loading CSM model from {model_id}...")
        print(f"[TTSS] NOTE: If you get a 401/403 error, run: huggingface-cli login")
        
        try:
            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_id)
            model = CsmForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
                raise RuntimeError(
                    f"[TTSS] CSM model access denied! This is a GATED model.\n"
                    f"Please run: huggingface-cli login\n"
                    f"Then visit https://huggingface.co/sesame/csm-1b and accept the terms.\n"
                    f"Original error: {e}"
                )
            raise
        
        # Build conversation with speaker ID
        # CSM uses [speaker_id] prefix in the text
        conversation = [
            {"role": "user", "content": f"[{speaker_id}]{text}"}
        ]
        
        # If context audio provided, add it as conversation history
        if context_audio and os.path.exists(context_audio):
            print(f"[TTSS] Using context audio: {context_audio}")
            context_waveform, context_sr = torchaudio.load(context_audio)
            # Resample to 24kHz if needed (CSM uses 24kHz)
            if context_sr != 24000:
                resampler = torchaudio.transforms.Resample(context_sr, 24000)
                context_waveform = resampler(context_waveform)
            # Add context as previous turn
            conversation.insert(0, {
                "role": "assistant", 
                "content": f"[{speaker_id}]",
                "audio": context_waveform.squeeze().numpy()
            })
        
        # Process inputs
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
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
class TTSSLoadReferenceAudio:
    """
    Load reference audio for voice cloning.
    Files should be placed in: models/tts/reference_audio/
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        files = get_reference_audio_files()
        return {
            "required": {
                "audio_file": (files, {"default": files[0] if files else "(none)"}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load_reference"
    CATEGORY = "🔊 TTSS"
    
    def load_reference(self, audio_file, custom_path=""):
        """Load reference audio file."""
        if custom_path and os.path.exists(custom_path):
            return (custom_path,)
        
        if audio_file == "(none)":
            return (None,)
        
        ref_path = os.path.join(tts_reference_path, audio_file)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"[TTSS] Reference audio not found: {ref_path}")
        
        return (ref_path,)


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
# Exports
# =============================================================================
__all__ = [
    "TTSSTextToSpeech",
    "TTSSVoiceSelector", 
    "TTSSLoadReferenceAudio",
    "TTSSLoadSRT",
    "TTSSLoadAudio",
    "TTSSPreviewAudio",
    "TTSSCombineAudio",
    "TTSSSaveAudio",
]
