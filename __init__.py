"""
TTSS - Text-to-Speech
ComfyUI custom nodes for text-to-speech synthesis.

Supports multiple TTS engines:
- pyttsx3: Offline system voices (SAPI/NSSpeech/espeak)
- edge-tts: Microsoft Edge TTS (online, high quality, free)
- coqui-tts: Neural TTS with voice cloning (local GPU)

Model directory: ComfyUI/models/tts/
"""

import os
import sys

# Add current directory to path for imports
now_dir = os.path.dirname(os.path.abspath(__file__))
if now_dir not in sys.path:
    sys.path.insert(0, now_dir)

# Web directory for frontend extensions
WEB_DIRECTORY = "./web"

# Import nodes
from .nodes import (
    TTSSTextToSpeech,
    TTSSVoiceSelector,
    TTSSLoadReferenceAudio,
    TTSSLoadAudio,
    TTSSLoadSRT,
    TTSSPreviewAudio,
    TTSSCombineAudio,
    TTSSSaveAudio,
)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TTSSTextToSpeech": TTSSTextToSpeech,
    "TTSSVoiceSelector": TTSSVoiceSelector,
    "TTSSLoadReferenceAudio": TTSSLoadReferenceAudio,
    "TTSSLoadAudio": TTSSLoadAudio,
    "TTSSLoadSRT": TTSSLoadSRT,
    "TTSSPreviewAudio": TTSSPreviewAudio,
    "TTSSCombineAudio": TTSSCombineAudio,
    "TTSSSaveAudio": TTSSSaveAudio,
}

# Human-readable node names with prefix for search
NODE_DISPLAY_NAME_MAPPINGS = {
    "TTSSTextToSpeech": "🔊 TTSS Text to Speech",
    "TTSSVoiceSelector": "🎤 TTSS Voice Selector",
    "TTSSLoadReferenceAudio": "🎙️ TTSS Load Reference Audio",
    "TTSSLoadAudio": "📂 TTSS Load Audio",
    "TTSSLoadSRT": "📄 TTSS Load SRT",
    "TTSSPreviewAudio": "🎧 TTSS Preview Audio",
    "TTSSCombineAudio": "🔗 TTSS Combine Audio",
    "TTSSSaveAudio": "💾 TTSS Save Audio",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
