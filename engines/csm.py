"""
CSM TTS Engine
Conversational Speech Model (1B params, conversational, GPU)
"""

import os
import tempfile


def synth_csm(text, output_file, speaker_id, context_audio, 
              model_manager, tts_csm_path, csm_model_id, output_path):
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
        model_id = csm_model_id
        
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
            processor, model = model_manager.get_csm(local_model_path, device)
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
                model_manager.unload_csm(local_model_path, device)
            except Exception:
                pass


def synth_csm_conversation(conversation, output_file, model_manager, 
                          tts_csm_path, csm_model_id):
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
    
    model_id = csm_model_id
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
        processor, model = model_manager.get_csm(local_model_path, device)
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
            model_manager.unload_csm(local_model_path, device)
        except Exception:
            pass
