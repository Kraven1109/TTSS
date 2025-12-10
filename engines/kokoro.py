"""
Kokoro TTS Engine
Lightweight neural TTS (82M params, fast, multi-language)
"""

import os


def synth_kokoro(text, output_file, voice, lang_code, speed, keep_models, 
                 model_manager, tts_models_path, kokoro_repo_id):
    """Synthesize using Kokoro TTS via ONNX Runtime (lightweight 82M neural TTS).
    
    Uses kokoro-onnx package which works on Python 3.10-3.13.
    First run will download ~300MB model files.
    """
    import numpy as np
    
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
        download_kokoro_models(model_path, voices_path, kokoro_repo_id)
    
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
    kokoro = model_manager.get_kokoro(model_path, voices_path)
    
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
    import soundfile as sf
    sf.write(output_file, final_audio, sample_rate)
    
    # Decide final keep_models policy (per-node overrides global)
    effective_keep = bool(keep_models) or getattr(model_manager, 'keep_models_default', False)
    # Unload Kokoro if not expected to keep the model
    if not effective_keep:
        try:
            model_manager.unload_kokoro(model_path, voices_path)
        except Exception:
            pass


def download_kokoro_models(model_path, voices_path, kokoro_repo_id):
    """Download Kokoro ONNX model files using HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "[TTSS] huggingface_hub not installed. Run: pip install huggingface-hub"
        )
    
    repo_id = kokoro_repo_id
    
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
