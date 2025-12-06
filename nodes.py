"""
TTSS ComfyUI Nodes
Text-to-Speech nodes with multiple engine support.
Supports: pyttsx3 (offline), edge-tts (Microsoft), Coqui TTS (neural).
"""

import os
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
tts_xtts_path = os.path.join(tts_models_path, "xtts")  # For XTTS-v2 / Auralis models
tts_voices_path = os.path.join(tts_models_path, "voices")

for path in [output_path, tts_models_path, tts_reference_path, tts_xtts_path, tts_voices_path]:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# TTS Engine Registry
# =============================================================================
TTS_ENGINES = ["pyttsx3", "edge-tts", "xtts-v2"]

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

def get_xtts_models():
    """Get available XTTS models (Auralis-compatible)."""
    models = []
    
    # Check for local converted models in xtts folder
    xtts_path = os.path.join(tts_models_path, "xtts")
    if os.path.exists(xtts_path):
        for item in os.listdir(xtts_path):
            item_path = os.path.join(xtts_path, item)
            if os.path.isdir(item_path):
                # Look for safetensors (Auralis format)
                if any(f.endswith('.safetensors') for f in os.listdir(item_path)):
                    models.append(item)
    
    # Default HuggingFace models (Auralis)
    hf_models = [
        "AstraMindAI/xttsv2",  # Auralis default
    ]
    return models + hf_models if models else hf_models

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
    - edge-tts: Microsoft Edge TTS (online, high quality, free)
    - xtts-v2: Neural TTS with voice cloning via Auralis (GPU, Python 3.10+)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get voices for each engine
        pyttsx3_voices = get_pyttsx3_voices()
        edge_voices = get_edge_tts_voices()
        xtts_models = get_xtts_models()
        
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
                "xtts_model": (xtts_models, {"default": xtts_models[0] if xtts_models else "AstraMindAI/xttsv2"}),
            },
            "optional": {
                "text_input": ("STRING", {"forceInput": True, "multiline": True}),
                "srt_input": ("SRT",),
                "reference_audio": ("AUDIOPATH",),
            }
        }
    
    RETURN_TYPES = ("AUDIOPATH",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "synthesize"
    CATEGORY = "🔊 TTSS"
    
    def synthesize(self, text, engine, speed, 
                   pyttsx3_voice="default", edge_voice="en-US-AriaNeural",
                   xtts_model="AstraMindAI/xttsv2", reference_audio="(none)",
                   text_input=None, srt_input=None):
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
        elif engine == "xtts-v2":
            voice_name = xtts_model
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
        elif engine == "xtts-v2":
            self._synth_xtts(final_text, output_file, voice_name, ref_audio_path)
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
    
    def _synth_pyttsx3(self, text, output_file, voice_name, speed):
        """Synthesize using pyttsx3 (offline system voices)."""
        import pyttsx3
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
    
    def _synth_xtts(self, text, output_file, model_name, reference_audio):
        """Synthesize using XTTS-v2 via Auralis (Python 3.10+ compatible)."""
        try:
            from auralis import TTS, TTSRequest
        except ImportError:
            raise ImportError(
                "[TTSS] Auralis not installed. Run: pip install auralis\n"
                "Auralis provides XTTS-v2 support for Python 3.10+ (unlike Coqui TTS which requires Python <3.12)"
            )
        
        # Initialize TTS model
        model = model_name if model_name else "AstraMindAI/xttsv2"
        
        # Check if local path or HuggingFace model
        xtts_local_path = os.path.join(tts_models_path, "xtts", model_name) if model_name else None
        
        if xtts_local_path and os.path.exists(xtts_local_path):
            # Local converted model
            tts = TTS().from_pretrained(xtts_local_path)
        else:
            # HuggingFace model
            tts = TTS().from_pretrained(model, gpt_model='AstraMindAI/xtts2-gpt')
        
        # Create TTS request
        request_kwargs = {"text": text}
        
        # Voice cloning with reference audio (required for XTTS)
        if reference_audio and os.path.exists(reference_audio):
            request_kwargs["speaker_files"] = [reference_audio]
        else:
            # XTTS requires reference audio for voice cloning
            # Use a default reference if available
            default_ref = os.path.join(tts_reference_path, "default.wav")
            if os.path.exists(default_ref):
                request_kwargs["speaker_files"] = [default_ref]
            else:
                raise ValueError(
                    "[TTSS] XTTS-v2 requires reference audio for voice cloning.\n"
                    "Please provide a reference audio file or place a 'default.wav' in:\n"
                    f"  {tts_reference_path}"
                )
        
        request = TTSRequest(**request_kwargs)
        
        # Generate speech
        output = tts.generate_speech(request)
        output.save(output_file)
    
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
