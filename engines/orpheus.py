"""
Orpheus TTS Engine
SOTA LLM-based TTS with emotion tags (3B params, GPU)
"""

import os


def synth_orpheus(text, output_file, lang, voice, keep_models, 
                  model_manager, tts_orpheus_path, orpheus_models, snac_repo_id):
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
    
    # Default to English model if language not found
    orpheus_repo = orpheus_models.get(lang, orpheus_models.get("English", "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"))
    
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
            repo_id=snac_repo_id,
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
    orpheus = model_manager.get_orpheus(lang, n_gpu_layers=n_gpu_layers)

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
    effective_keep = bool(keep_models) or getattr(model_manager, 'keep_models_default', False)
    # Unload the orpheus instance if configured to not keep models loaded
    if not effective_keep:
        try:
            model_manager.unload_orpheus(lang, n_gpu_layers=n_gpu_layers)
        except Exception:
            pass
    
    # If no audio generated, raise
    if len(audio) == 0:
        raise RuntimeError("[TTSS] Orpheus generated no audio")
