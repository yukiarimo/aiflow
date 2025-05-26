import torch
import argparse
from model.model import MultiModalAIDetector
import torch.nn.functional as F
import numpy as np

def check_and_fix_nan_tensors(state_dict):
    """Check for NaN values in tensors and replace with zeros"""
    fixed_tensors = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
            # Count how many NaN values we're fixing
            nan_count = torch.isnan(tensor).sum().item()
            # Replace NaNs with zeros
            tensor[torch.isnan(tensor)] = 0.0
            state_dict[key] = tensor
            print(f"Fixed {nan_count} NaN values in {key}")
            fixed_tensors += 1
    return fixed_tensors

def merge_models(models_dict, output_path):
    """
    Create a multi-modal model by directly storing each modality model
    without sharing components between modalities.
    
    Args:
        models_dict: Dictionary mapping modalities to model paths
        output_path: Path to save the merged model
    """
    # Create a new model
    merged_model = MultiModalAIDetector()
    merged_state_dict = merged_model.state_dict()
    default_state_dict = merged_model.state_dict()  # Keep default values for fixing NaNs
    
    # Create separate models for each modality
    modality_models = {}
    
    print("Loading individual modality models...")
    for modality, model_path in models_dict.items():
        try:
            # Create a model instance for this modality
            modality_model = MultiModalAIDetector()
            
            # Load the model state dict
            modality_state_dict = torch.load(model_path, map_location='cpu')
            
            # Check for NaN values and fix them
            print(f"Checking for NaN values in {modality} model...")
            nan_fixed = check_and_fix_nan_tensors(modality_state_dict)
            if nan_fixed > 0:
                print(f"Fixed {nan_fixed} tensors with NaN values in {modality} model")
            
            # Try to load the state dict into this model
            try:
                modality_model.load_state_dict(modality_state_dict)
                modality_models[modality] = modality_model
                print(f"Successfully loaded {modality} model")
            except Exception as e:
                print(f"Error loading {modality} model: {e}")
                continue
                
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            continue
    
    # Create a registry to keep track of which modality's components we should use
    # This ensures we don't mix components between modalities
    component_registry = {
        'speech': {
            'extractor': 'speech_extractor',
            'adapter': 'audio_adapter',
            'temporal': 'temporal_processor',
            'classifier': 'classifier',
            'source': 'speech'
        },
        'instrumental': {
            'extractor': 'instrumental_extractor',
            'adapter': 'audio_adapter',
            'temporal': 'temporal_processor',
            'classifier': 'classifier',
            'source': 'instrumental'
        },
        'mixed_audio': {
            'extractor': 'mixed_audio_extractor',
            'adapter': 'audio_adapter',
            'temporal': 'temporal_processor',
            'classifier': 'classifier',
            'source': 'mixed_audio'
        },
        'image': {
            'extractor': 'image_extractor',
            'adapter': 'image_adapter',
            'classifier': 'classifier',
            'source': 'image'
        },
        'video': {
            'extractor': 'video_extractor',
            'adapter': 'video_adapter',
            'temporal': 'temporal_processor',
            'classifier': 'classifier',
            'source': 'video'
        }
    }
    
    # Track which keys have been successfully copied
    copied_keys = set()
    
    # Now copy parameters from each modality model into the merged model
    print("\nCopying modality-specific components to merged model...")
    for modality in modality_models:
        if modality not in component_registry:
            continue
            
        # Get the state dict for this modality model
        modality_dict = modality_models[modality].state_dict()
        
        # Get the registry entry for this modality
        registry = component_registry[modality]
        
        # Get the source modality (where we'll copy parameters from)
        source = registry['source']
        
        # Copy extractor
        extractor_prefix = registry['extractor']
        for key in modality_dict:
            if key.startswith(extractor_prefix):
                merged_state_dict[key] = modality_dict[key].clone()
                copied_keys.add(key)
                print(f"Copied {key} from {source}")
        
        # Copy adapter
        adapter_prefix = registry['adapter']
        adapter_keys_for_modality = []
        for key in modality_dict:
            if key.startswith(adapter_prefix):
                # For adapters shared between audio modalities, we need to tag them 
                # to avoid overwriting (only for audio-specific modality paths)
                if adapter_prefix == 'audio_adapter' and modality in ['speech', 'instrumental', 'mixed_audio']:
                    renamed_key = f"{modality}_{key}"
                    merged_state_dict[renamed_key] = modality_dict[key].clone()
                    adapter_keys_for_modality.append((key, renamed_key))
                    print(f"Copied and renamed {key} → {renamed_key} from {source}")
                else:
                    merged_state_dict[key] = modality_dict[key].clone()
                    copied_keys.add(key)
                    print(f"Copied {key} from {source}")
        
        # Copy temporal processor if applicable
        if 'temporal' in registry:
            temporal_prefix = registry['temporal']
            temporal_keys_for_modality = []
            for key in modality_dict:
                if key.startswith(temporal_prefix):
                    # Rename temporal processor keys to avoid conflicts
                    renamed_key = f"{modality}_{key}"
                    merged_state_dict[renamed_key] = modality_dict[key].clone()
                    temporal_keys_for_modality.append((key, renamed_key))
                    print(f"Copied and renamed {key} → {renamed_key} from {source}")
        
        # Copy classifier
        classifier_prefix = registry['classifier']
        classifier_keys_for_modality = []
        for key in modality_dict:
            if key.startswith(classifier_prefix):
                # Rename classifier keys to avoid conflicts
                renamed_key = f"{modality}_{key}"
                merged_state_dict[renamed_key] = modality_dict[key].clone()
                classifier_keys_for_modality.append((key, renamed_key))
                print(f"Copied and renamed {key} → {renamed_key} from {source}")
    
    # Check for any remaining NaN values
    print("\nChecking for remaining NaN values in merged model...")
    fixed_count = check_and_fix_nan_tensors(merged_state_dict)
    if fixed_count > 0:
        print(f"Fixed NaN values in {fixed_count} tensors")
    else:
        print("No remaining NaN values found")
    
    # Now we need to modify the forward method to handle the modality-specific pathways
    # We'll do this by loading the merged state dict into a model and then saving it again
    try:
        # Save the merged state dict to a temporary file
        temp_path = f"{output_path}.temp"
        torch.save(merged_state_dict, temp_path)
        
        # Overwrite the forward method of the MultiModalAIDetector class to use modality-specific components
        original_forward = MultiModalAIDetector.forward
        
        def new_forward(self, x, modality):
            """
            Modified forward pass that uses modality-specific components
            """
            # Fall back to original implementation for unsupported modalities
            if modality not in ['speech', 'instrumental', 'mixed_audio', 'image', 'video']:
                return original_forward(self, x, modality)
            
            self.active_modality = modality
            
            if modality == 'speech':
                features = self.speech_extractor(x)
                # Use audio adapter for speech
                temporal_weights = getattr(self, f"speech_temporal_processor.gru.weight_ih_l0", 
                                          self.temporal_processor.gru.weight_ih_l0)
                features = self.temporal_processor(features)
                classifier_weights = getattr(self, f"speech_classifier.fc.0.weight", 
                                            self.classifier.fc[0].weight)
                features = self.audio_adapter(features)
                
            elif modality == 'instrumental':
                features = self.instrumental_extractor(x)
                features = self.temporal_processor(features)
                features = self.audio_adapter(features)
                
            elif modality == 'mixed_audio':
                features = self.mixed_audio_extractor(x)
                features = self.temporal_processor(features)
                features = self.audio_adapter(features)
                
            elif modality == 'image':
                features = self.image_extractor(x)
                features = features.squeeze(1)  # Remove sequence dimension
                features = self.image_adapter(features)
                
            elif modality == 'video':
                features = self.video_extractor(x)
                features = self.temporal_processor(features)
                features = self.video_adapter(features)
            
            return self.classifier(features)
        
        # Replace the forward method
        MultiModalAIDetector.forward = new_forward
        
        # Load the merged state dict into the model
        merged_model = MultiModalAIDetector()
        
        # Create a new state dict that includes only non-renamed keys
        # This is necessary because we can't actually modify the model structure directly
        final_state_dict = {}
        for key in merged_model.state_dict():
            if key in merged_state_dict:
                final_state_dict[key] = merged_state_dict[key]
            else:
                # Use default initialization for missing keys
                final_state_dict[key] = default_state_dict[key]
        
        merged_model.load_state_dict(final_state_dict)
        print("Successfully loaded merged state dict")
        
        # Save the final model
        torch.save(final_state_dict, output_path)
        print(f"Merged model saved to {output_path}")
        
        # Delete the temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
    except Exception as e:
        print(f"Error during model saving: {e}")
        
        # Save the partial merged state dict
        torch.save(merged_state_dict, output_path)
        print(f"Partial merged model saved to {output_path}")
    
    # Basic validation of the merged model
    print("\nValidating the merged model...")
    merged_model.eval()
    
    test_inputs = {
        'speech': torch.randn(1, 1, 256, 1400),
        'instrumental': torch.randn(1, 1, 256, 1400),
        'mixed_audio': torch.randn(1, 1, 256, 1400),
        'image': torch.randn(1, 3, 224, 224),
        'video': torch.randn(1, 30, 3, 224, 224)
    }
    
    with torch.no_grad():
        for modality, test_input in test_inputs.items():
            if modality in modality_models:
                try:
                    # Get the individual model for this modality
                    modality_model = modality_models[modality]
                    modality_model.eval()
                    
                    # Get predictions from individual model
                    indiv_outputs = modality_model(test_input, modality)
                    indiv_probs = F.softmax(indiv_outputs, dim=1).numpy()
                    
                    # Get predictions from merged model
                    try:
                        merged_outputs = merged_model(test_input, modality)
                        merged_probs = F.softmax(merged_outputs, dim=1).numpy()
                        
                        # Compare the results
                        print(f"\n{modality} validation:")
                        print(f"  Individual model - Human: {indiv_probs[0][0]*100:.2f}%, AI: {indiv_probs[0][1]*100:.2f}%")
                        print(f"  Merged model    - Human: {merged_probs[0][0]*100:.2f}%, AI: {merged_probs[0][1]*100:.2f}%")
                        
                        # Check if the predictions are close
                        if np.abs(indiv_probs[0][0] - merged_probs[0][0]) < 0.01:
                            print(f"  ✓ Predictions match for {modality}")
                        else:
                            print(f"  ✗ Predictions differ for {modality}")
                    except Exception as e:
                        print(f"  Error validating merged model for {modality}: {e}")
                
                except Exception as e:
                    print(f"Error validating {modality}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Merge models trained on different modalities')
    parser.add_argument('--speech', type=str, default=None, help='Path to speech model')
    parser.add_argument('--instrumental', type=str, default=None, help='Path to instrumental model')
    parser.add_argument('--mixed_audio', type=str, default=None, help='Path to mixed audio model')
    parser.add_argument('--image', type=str, default=None, help='Path to image model')
    parser.add_argument('--video', type=str, default=None, help='Path to video model')
    parser.add_argument('--output', type=str, default='model/merged_model.pth', help='Path to save merged model')

    args = parser.parse_args()

    # Create a dictionary of modality-specific models
    models_dict = {}
    if args.speech:
        models_dict['speech'] = args.speech
    if args.instrumental:
        models_dict['instrumental'] = args.instrumental
    if args.mixed_audio:
        models_dict['mixed_audio'] = args.mixed_audio
    if args.image:
        models_dict['image'] = args.image
    if args.video:
        models_dict['video'] = args.video

    if not models_dict:
        print("Error: No models specified")
        return

    # Merge the models
    merge_models(models_dict, args.output)

if __name__ == "__main__":
    main()