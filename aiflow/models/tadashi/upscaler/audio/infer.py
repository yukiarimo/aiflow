import torch
import torchaudio
import json
import os
from model import AudioUpsampler
from utils import load_audio, process_in_chunks # Use updated process_in_chunks

def infer(config_path):
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
         print(f"Error: Configuration file not found at {config_path}")
         return
    except json.JSONDecodeError:
         print(f"Error: Could not parse configuration file {config_path}")
         return

    # --- Extract relevant config values ---
    cfg_audio = config['audio']
    cfg_model = config['model'] # Use model config potentially from checkpoint later
    cfg_infer = config['inference']

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for inference.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Warning: MPS not found, using CUDA device for inference.")
    else:
        device = torch.device("cpu")
        print("Warning: MPS or CUDA not found, using CPU device for inference.")

    # --- Load Model ---
    checkpoint_path = cfg_infer['model_checkpoint']
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
        print(f"Checkpoint loaded from {checkpoint_path}")

        # Load model config from checkpoint if available, otherwise use current config
        # Prioritize checkpoint's config for model architecture reconstruction
        if 'config' in checkpoint:
             chkpt_cfg_model = checkpoint['config']['model']
             chkpt_cfg_audio = checkpoint['config']['audio']
             print("Using model architecture and audio settings from checkpoint.")
        else:
             # Fallback to current config if checkpoint doesn't store it
             chkpt_cfg_model = cfg_model
             chkpt_cfg_audio = cfg_audio
             print("Warning: Checkpoint does not contain config, using current config for model architecture.")

        upsample_factor = chkpt_cfg_audio['target_sr'] // chkpt_cfg_audio['low_res_sr']
        model = AudioUpsampler(
            in_channels=chkpt_cfg_audio['channels'],
            out_channels=chkpt_cfg_audio['channels'],
            base_channels=chkpt_cfg_model['base_channels'],
            depth=chkpt_cfg_model['depth'],
            upsample_factor=upsample_factor
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) # Move model to target device
        model.eval() # Set model to evaluation mode
        print(f"Model weights loaded successfully (trained for epoch {checkpoint.get('epoch', 'N/A')}).")

    except Exception as e:
         print(f"Error loading model checkpoint: {e}")
         return

    # --- Load Input Audio ---
    input_file = cfg_infer['input_file']
    low_res_sr = chkpt_cfg_audio['low_res_sr'] # Use SR from model training settings
    target_sr = chkpt_cfg_audio['target_sr']   # Use SR from model training settings

    if not os.path.exists(input_file):
         print(f"Error: Input audio file not found: {input_file}")
         return

    print(f"Loading input audio: {input_file} (Expecting {low_res_sr} Hz)")
    # Load audio, ensuring utils resamples it *to* low_res_sr if it isnt already
    waveform_lr = load_audio(input_file, low_res_sr)
    if waveform_lr is None:
        print(f"Failed to load input audio: {input_file}")
        return

    print(f"Input audio shape: {waveform_lr.shape}")

    # --- Inference ---
    print("Performing inference...")
    try:
        with torch.no_grad():
            # Decide whether to use chunking based on length
            inference_chunk_size_samples_lr = int(low_res_sr * config['audio']['inference_chunk_size_ms'] / 1000)
            # Use chunking if audio is longer than the inference chunk size
            if waveform_lr.shape[1] > inference_chunk_size_samples_lr:
                print(f"Input is long, using chunked inference (LR chunk samples: {inference_chunk_size_samples_lr})")
                output_waveform_hr = process_in_chunks(
                    model=model,
                    waveform_lr=waveform_lr, # Shape [C, L_low]
                    chunk_len_lr=inference_chunk_size_samples_lr,
                    upsample_factor=upsample_factor,
                    device=device,
                    overlap_ratio=0.5 # Can adjust overlap_ratio
                )
            else:
                print("Input is short, processing in one go.")
                # Add batch dimension, send to device, run model, remove batch dim, move to CPU
                waveform_lr = waveform_lr.unsqueeze(0).to(device) # [1, C, L_low]
                output_waveform_hr = model(waveform_lr)           # [1, C, L_high]
                output_waveform_hr = output_waveform_hr.squeeze(0).cpu() # [C, L_high]

        print(f"Output audio shape: {output_waveform_hr.shape}")

        # --- Save Output Audio ---
        output_file = cfg_infer['output_file']
        output_dir = os.path.dirname(output_file)
        if output_dir: # Create output directory if specified and doesn't exist
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving output audio to: {output_file} at {target_sr} Hz")
        torchaudio.save(output_file, output_waveform_hr, target_sr)

        print("Inference complete.")

    except Exception as e:
        print(f"An error occurred during inference or saving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Audio Upsampling Inference")
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file (JSON)')
    parser.add_argument('--input', type=str, help='Path to the low-resolution input audio file (overrides config)')
    parser.add_argument('--output', type=str, help='Path to save the high-resolution output audio file (overrides config)')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint file (overrides config)')
    args = parser.parse_args()

    temp_config_path = None # Flag for temporary config file
    try:
         # Update config with command-line overrides if provided
        if args.input or args.output or args.checkpoint:
            with open(args.config, 'r') as f:
                config_data = json.load(f)

            if args.input:
                config_data['inference']['input_file'] = args.input
            if args.output:
                config_data['inference']['output_file'] = args.output
            if args.checkpoint:
                config_data['inference']['model_checkpoint'] = args.checkpoint

            # Use a temporary file for the modified config
            temp_config_path = "temp_infer_config_override.json"
            with open(temp_config_path, 'w') as f:
                 json.dump(config_data, f, indent=2)
            print(f"Using temporary config with overrides: {temp_config_path}")
            infer(temp_config_path)
        else:
           # Use the config file directly if no overrides
           infer(args.config)

    finally:
         # Clean up temporary config file if it was created
         if temp_config_path and os.path.exists(temp_config_path):
             os.remove(temp_config_path)
             print(f"Removed temporary config file: {temp_config_path}")