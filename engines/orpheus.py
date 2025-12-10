"""
Orpheus TTS Engine
SOTA LLM-based TTS with emotion tags (3B params, GPU)
Uses Transformers with optional Unsloth acceleration for easier maintenance
"""

import os
import re


def synth_orpheus(text, output_file, lang, voice, keep_models, 
                  model_manager, tts_orpheus_path, orpheus_model_id, snac_repo_id):
    """Synthesize using Orpheus TTS via Unsloth/Transformers (Python backend).
    
    Replaces llama.cpp backend with Pure Python stack for easier maintenance.
    Works on Windows/Linux/macOS without C++ compilation.
    
    Features:
    - Uses Transformers with optional Unsloth acceleration
    - Automatic fallback to fp16 if bitsandbytes unavailable
    - Proper SNAC code extraction from decoded text
    - Supports inline emotion tags: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
    
    Requirements:
    - pip install snac transformers torchaudio accelerate
    - Optional: pip install unsloth bitsandbytes (for acceleration and 4-bit quantization)
    
    Args:
        text: Input text to synthesize
        output_file: Path to save WAV output
        lang: Language (English, French, German, Korean, Hindi, Mandarin, Spanish, Italian)
        voice: Voice identifier (e.g., "en_tara", "fr_speaker_0")
        keep_models: Keep models loaded in memory for faster subsequent runs
    """
    import torch
    import numpy as np
    from scipy.io.wavfile import write as wav_write
    
    try:
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torchaudio
    except ImportError as e:
        raise ImportError(
            "[TTSS] Missing dependencies for Orpheus TTS.\n"
            "Run: pip install snac transformers torchaudio accelerate\n"
            f"Missing: {e}"
        )

    # Check if bitsandbytes is available for 4-bit quantization
    try:
        import bitsandbytes
        has_bitsandbytes = True
        print("[TTSS] bitsandbytes detected - will use 4-bit quantization")
    except ImportError:
        has_bitsandbytes = False
        print("[TTSS] bitsandbytes not available - using fp16 (requires more VRAM)")

    # Get Model ID - use unified model for all languages
    model_id = orpheus_model_id
    print(f"[TTSS] Loading Orpheus model: {model_id}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[TTSS] WARNING: No CUDA detected. Orpheus on CPU will be extremely slow!")
    
    # Load Model & Tokenizer
    model = None
    tokenizer = None
    
    # Try Unsloth first (optional acceleration)
    try:
        from unsloth import FastLanguageModel
        if has_bitsandbytes:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                dtype=None, 
                load_in_4bit=True,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=2048,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        FastLanguageModel.for_inference(model)
        print("[TTSS] Loaded with Unsloth acceleration")
    except (ImportError, Exception) as e:
        # ImportError: Unsloth not installed
        # Other exceptions: Unsloth failed to load model
        if not isinstance(e, ImportError):
            print(f"[TTSS] Unsloth failed ({e}), falling back to standard Transformers...")
        else:
            print("[TTSS] Unsloth not found, using standard Transformers...")
        model = None
    
    # Fall back to standard Transformers if Unsloth failed
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if has_bitsandbytes and device == "cuda":
            print("[TTSS] Loading in 4-bit mode...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16,
            )
        else:
            print("[TTSS] Loading in fp16/fp32 mode (no quantization)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )

    # Load SNAC Decoder
    try:
        snac_device = device if device == "cuda" else "cpu"
        snac_model = SNAC.from_pretrained(snac_repo_id).to(snac_device)
    except Exception as e:
        raise RuntimeError(f"[TTSS] Failed to load SNAC decoder: {e}")

    # Format Prompt (CRITICAL - Orpheus expects this exact format)
    prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
    
    # Tokenize - when using device_map="auto", the model handles device placement
    # So we don't manually move inputs to a specific device
    if hasattr(model, 'device'):
        # Model is on a specific device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    else:
        # Model uses device_map, let it handle placement
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move to first available device if model has multiple devices
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_device = list(model.hf_device_map.values())[0]
            inputs = inputs.to(first_device)

    # Generate Tokens
    print("[TTSS] Generating Orpheus tokens...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1
        )
    
    # Decode to text and extract SNAC codes
    # The model outputs text containing SNAC codes in format like: [1,2,3][4,5,6]...
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"[TTSS] Decoded output preview: {decoded_text[:200]}...")
    
    # Extract SNAC codes from the decoded text
    snac_codes = _extract_snac_codes(decoded_text)
    
    if snac_codes is None or len(snac_codes) == 0:
        raise RuntimeError("[TTSS] Failed to extract SNAC codes from model output")
    
    print(f"[TTSS] Extracted {len(snac_codes)} SNAC code groups")
    
    # Decode SNAC codes to audio
    print("[TTSS] Decoding SNAC codes to audio...")
    try:
        # Convert SNAC codes to tensor format expected by SNAC decoder
        # SNAC decode() expects shape: [num_hierarchies, sequence_length] (NO batch dimension)
        # snac_codes from _extract_snac_codes is already [num_hierarchies, sequence_length]
        snac_tensor = torch.tensor(snac_codes, dtype=torch.long).to(snac_device)
        print(f"[TTSS] SNAC tensor shape: {snac_tensor.shape} (expected: [num_hierarchies=3, sequence_length])")
        
        # Verify we have the correct shape
        if snac_tensor.ndim != 2:
            raise RuntimeError(f"[TTSS] SNAC codes have wrong dimensions: {snac_tensor.shape}, expected 2D [hierarchies, sequence]")
        if snac_tensor.shape[0] != 3:
            raise RuntimeError(f"[TTSS] SNAC codes have {snac_tensor.shape[0]} hierarchies, expected 3")
        
        # Decode to audio waveform
        with torch.no_grad():
            audio_output = snac_model.decode(snac_tensor)
        
        # Convert to numpy and ensure proper shape
        audio_np = audio_output.cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()
        
        # Normalize to [-1, 1] range
        max_abs = np.abs(audio_np).max()
        if max_abs > 0:
            audio_np = audio_np / max_abs
        
        # Convert to int16 for WAV output
        audio_int16 = (audio_np * 32767.0).astype(np.int16)
        
        # Save as WAV file (SNAC uses 24kHz)
        sample_rate = 24000
        wav_write(output_file, sample_rate, audio_int16)
        
        print(f"[TTSS] Audio saved: {output_file}")
        
    except Exception as e:
        raise RuntimeError(f"[TTSS] Failed to decode SNAC codes to audio: {e}")
    
    # Clean up models if not keeping them loaded
    if not keep_models == "True":
        try:
            del model
            del tokenizer
            del snac_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _extract_snac_codes(decoded_text):
    """
    Extract SNAC codes from the decoded model output.
    
    Handles both bracketed format [1,2,3,4,5,6,7] and raw numbers.
    SNAC expects shape: [num_hierarchies, sequence_length]
    
    Args:
        decoded_text: The decoded text from the model
        
    Returns:
        numpy array of SNAC codes with shape [num_hierarchies, sequence_length] or None if extraction failed
    """
    import numpy as np
    import re

    # Debug: Print first 500 chars of decoded text to understand format
    print(f"[TTSS] Decoded text preview: {decoded_text[:500]}...")
    
    # Pattern 1: Bracketed SNAC codes [1,2,3,4,5,6,7][8,9,10,11,12,13,14]...
    matches = re.findall(r'\[([^\]]+)\]', decoded_text)
    if matches:
        print(f"[TTSS] Found {len(matches)} bracketed matches")
        try:
            frames = []
            for match in matches:
                codes = [int(c.strip()) for c in match.split(',') if c.strip().lstrip('-').isdigit()]
                if codes:
                    frames.append(codes)
            if frames:
                arr = np.array(frames)
                print(f"[TTSS] Extracted {len(frames)} frames with shape {arr.shape}")
                return arr.T  # [num_hierarchies, sequence_length]
        except Exception as e:
            print(f"[TTSS] Bracket parse failed: {e}")

    # Pattern 2: Raw numbers fallback
    numbers = re.findall(r'-?\d+', decoded_text)
    print(f"[TTSS] Found {len(numbers)} raw numbers")
    if numbers:
        print(f"[TTSS] First 20 numbers: {numbers[:20]}")
        try:
            codes = [int(n) for n in numbers]
            
            # SNAC validation: SNAC uses 3 codebooks, not 7!
            # Codes should be in valid range (typically 0-4095 for 12-bit codes)
            max_valid_code = 4095
            invalid_codes = [c for c in codes if c > max_valid_code or c < 0]
            if invalid_codes:
                print(f"[TTSS] WARNING: Found {len(invalid_codes)} codes outside valid SNAC range [0-{max_valid_code}]")
                print(f"[TTSS] Invalid codes: {invalid_codes[:10]}...")
                # Clamp codes to valid range
                codes = [max(0, min(c, max_valid_code)) for c in codes]
                print(f"[TTSS] Clamped codes to valid range")
            
            # SNAC uses 3 codebooks, not 7!
            num_hierarchies = 3  # SNAC has 3 quantizers/codebooks
            num_frames = len(codes) // num_hierarchies
            print(f"[TTSS] Can make {num_frames} frames with {num_hierarchies} SNAC codebooks")
            if num_frames > 0:
                arr = np.array(codes[:num_frames * num_hierarchies]).reshape(num_frames, num_hierarchies).T
                print(f"[TTSS] Final SNAC array shape: {arr.shape}")
                print(f"[TTSS] SNAC code range: [{arr.min()}, {arr.max()}]")
                return arr
        except Exception as e:
            print(f"[TTSS] Fallback parse failed: {e}")

    print("[TTSS] Warning: No SNAC codes found")
    return None


