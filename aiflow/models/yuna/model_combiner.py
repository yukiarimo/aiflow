import torch
from safetensors.torch import load_file, save_file
from pathlib import Path
import argparse

def merge_model_weights(original_path, audio_path, trained_path, output_path, llm_ratio=0.7):
    """
    Merges three model checkpoints:
    1. Replaces vision weights from trained with original
    2. Keeps audio weights from trained
    3. Merges LLM weights (70% trained + 30% audio)
    4. Converts all weights to BF16
    
    Args:
        original_path: Path to original model (without audio)
        audio_path: Path to model with audio
        trained_path: Path to final trained model
        output_path: Path to save merged model
        llm_ratio: Ratio of trained model weights (default 0.7)
    """
    print("Loading checkpoints...")
    original_weights = load_file(original_path)
    audio_weights = load_file(audio_path)
    trained_weights = load_file(trained_path)
    
    merged_weights = {}
    
    print("\nProcessing weights...")
    
    for key, value in trained_weights.items():
        # 1. Replace all vision weights with original
        if key.startswith("visual."):
            if key in original_weights:
                merged_weights[key] = original_weights[key].to(torch.bfloat16)
                print(f"✓ Vision: {key} <- original")
            else:
                print(f"⚠ Warning: {key} not found in original, keeping trained")
                merged_weights[key] = value.to(torch.bfloat16)
        
        # 2. Keep audio weights from trained
        elif key.startswith("audio_encoder.") or key.startswith("audio_projector."):
            merged_weights[key] = value.to(torch.bfloat16)
            print(f"✓ Audio: {key} <- trained")
        
        # 3. Merge LLM weights (70% trained + 30% audio)
        elif key.startswith("model."):
            if key in audio_weights:
                trained_tensor = value.to(torch.float32)
                audio_tensor = audio_weights[key].to(torch.float32)
                
                # Weighted merge
                merged_tensor = (llm_ratio * trained_tensor + 
                               (1 - llm_ratio) * audio_tensor)
                
                merged_weights[key] = merged_tensor.to(torch.bfloat16)
                print(f"✓ LLM: {key} <- {int(llm_ratio*100)}% trained + {int((1-llm_ratio)*100)}% audio")
            else:
                print(f"⚠ Warning: {key} not found in audio model, using trained only")
                merged_weights[key] = value.to(torch.bfloat16)
        
        # Handle any other weights
        else:
            merged_weights[key] = value.to(torch.bfloat16)
            print(f"✓ Other: {key} <- trained")
    
    # Save merged weights
    print(f"\n💾 Saving merged model to {output_path}...")
    save_file(merged_weights, output_path)
    
    # Print statistics
    print("\n" + "="*60)
    print("Merge Statistics:")
    print("="*60)
    vision_count = sum(1 for k in merged_weights if k.startswith("visual."))
    audio_count = sum(1 for k in merged_weights if k.startswith("audio_"))
    llm_count = sum(1 for k in merged_weights if k.startswith("model."))
    other_count = len(merged_weights) - vision_count - audio_count - llm_count
    
    print(f"Vision weights (from original): {vision_count}")
    print(f"Audio weights (from trained): {audio_count}")
    print(f"LLM weights (merged): {llm_count}")
    print(f"Other weights: {other_count}")
    print(f"Total weights: {len(merged_weights)}")
    print(f"All weights converted to: BF16")
    print("="*60)
    print("✅ Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Yuna model checkpoints")
    parser.add_argument("--original", type=str, required=True, 
                       help="Path to original model safetensors (without audio)")
    parser.add_argument("--audio", type=str, required=True,
                       help="Path to model with audio safetensors")
    parser.add_argument("--trained", type=str, required=True,
                       help="Path to final trained model safetensors")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save merged model safetensors")
    parser.add_argument("--llm_ratio", type=float, default=0.7,
                       help="Ratio of trained model weights for LLM (default: 0.7)")
    
    args = parser.parse_args()
    
    merge_model_weights(
        original_path=args.original,
        audio_path=args.audio,
        trained_path=args.trained,
        output_path=args.output,
        llm_ratio=args.llm_ratio
    )

# --- EXAMPLE USAGE ---
"""
python model_combiner.py \
     --original '/Users/yuki/Documents/Github/yuna-ai/lib/models/yuna/original/model.safetensors' \
     --audio '/Users/yuki/Documents/Github/yuna-ai/lib/models/yuna/Yuna-2B-ft-audio/model.safetensors' \
     --trained '/Users/yuki/Downloads/converted_model_epoch2audio.safetensors' \
     --output 'Yuna-merged-model.safetensors' \
     --llm_ratio 0.7
"""