"""
pyttsx3 TTS Engine
Offline system voices (SAPI/NSSpeech/espeak)
"""


def synth_pyttsx3(text, output_file, voice_name, speed):
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
