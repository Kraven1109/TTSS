"""
edge-tts TTS Engine
Microsoft Edge TTS (online, high quality, free)
"""

import os
import subprocess
import threading
import shutil


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


def synth_edge_tts(text, output_file, voice_name, speed):
    """Synthesize using Microsoft Edge TTS via CLI (avoids async issues)."""
    voice = voice_name if voice_name else "en-US-AriaNeural"
    
    # Validate and clean text
    if not text or not text.strip():
        raise ValueError("[TTSS] Empty text provided to edge-tts")
    
    # edge-tts can be sensitive to certain characters - ensure clean text
    text = text.strip()
    
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
        print(f"[TTSS] edge-tts: Synthesizing with voice '{voice}', text length: {len(text)} chars")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            # Provide more helpful error message
            if "NoAudioReceived" in error_msg:
                raise RuntimeError(
                    f"edge-tts NoAudioReceived error. This usually indicates:\n"
                    f"  1. Network connectivity issues\n"
                    f"  2. Microsoft's TTS service is temporarily unavailable\n"
                    f"  3. Voice '{voice}' may not support this text\n"
                    f"  Try: Different voice, check network, or retry later.\n"
                    f"  Original error: {error_msg}"
                )
            raise RuntimeError(f"edge-tts failed: {error_msg}")
    except FileNotFoundError:
        # edge-tts CLI not in PATH, try Python API with thread
        _synth_edge_tts_threaded(text, output_file, voice, speed)


def _synth_edge_tts_threaded(text, output_file, voice, speed):
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
