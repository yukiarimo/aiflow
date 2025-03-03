import torch
import numpy as np
from model import EmotionalModel
from utils import get_text_embedding, load_config
import matplotlib.pyplot as plt
import time
import os

class EmotionalMemory:
    def __init__(self, max_length=1000):
        self.text = []
        self.max_length = max_length
        
    def update(self, new_text):
        self.text.append(new_text)
        # Trim if exceeds max length
        if len(self.text) > self.max_length:
            self.text = self.text[-self.max_length:]
    
    def get_full(self):
        return " ".join(self.text)
    
    def get_recent(self, n=5):
        return " ".join(self.text[-n:])

class EmotionalStateTracker:
    def __init__(self, emotion_names, window_size=10):
        self.emotion_names = emotion_names
        self.window_size = window_size
        self.history = {name: [] for name in emotion_names}
        
    def update(self, emotion_state):
        for name, value in emotion_state.items():
            self.history[name].append(value)
            # Keep only window_size recent values
            if len(self.history[name]) > self.window_size:
                self.history[name] = self.history[name][-self.window_size:]
                
    def get_smoothed_state(self):
        """Get exponentially smoothed emotional state"""
        result = {}
        for name in self.emotion_names:
            if not self.history[name]:
                result[name] = 0.0
            else:
                # More weight to recent values
                weights = np.exp(np.linspace(-2, 0, len(self.history[name])))
                weights /= weights.sum()  # Normalize
                result[name] = np.average(self.history[name], weights=weights)
        return result

def plot_emotions(tracker):
    """Create a visualization of the emotional state"""
    os.makedirs("emotion_plots", exist_ok=True)
    
    # Create radar chart of emotional state
    emotions = list(tracker.history.keys())
    values = [tracker.get_smoothed_state()[e] for e in emotions]
    
    # Normalize to [0, 1] for plotting
    values_norm = [(v + 1) / 2 for v in values]  # From [-1,1] to [0,1]
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(emotions), endpoint=False).tolist()
    values_norm.append(values_norm[0])  # Close the loop
    angles.append(angles[0])  # Close the loop
    emotions.append(emotions[0])  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values_norm, 'o-', linewidth=2)
    ax.fill(angles, values_norm, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), emotions[:-1])
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("Yuna's Emotional State", size=20)
    
    timestamp = int(time.time())
    plt.savefig(f"emotion_plots/emotional_state_{timestamp}.png")
    plt.close()

def inference(config, model_path=None):
    # Use the safe model name for finding the file
    if model_path is None:
        model_name = config.get("model_name", "Kokoro")
        safe_model_name = model_name.replace(" ", "_")
        model_path = f"checkpoints/{safe_model_name}_best_model.pth"
    
    device = config["device"]
    
    # Initialize model
    model = EmotionalModel(config).to(device)
    
    # Load the checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using untrained model")
    
    model.eval()

    # Initialize memory and state tracker
    emotion_names = config["emotion_names"]
    memory = EmotionalMemory()
    tracker = EmotionalStateTracker(emotion_names)
    
    # Start with initial neutral state
    memory.update("<|begin_of_text|>")
    print(f"Initial Memory: '{memory.get_recent()}'")

    print("\nEnter text to interact with Yuna. Type 'exit' to quit, 'plot' to visualize emotions.")

    with torch.no_grad():
        while True:
            text_input = input("You: ")
            if text_input.lower() == "exit":
                break
            
            if text_input.lower() == "plot":
                plot_emotions(tracker)
                print("Emotional state visualization saved to emotion_plots folder")
                continue

            # Get text embedding
            input_embedding_np = get_text_embedding(text_input, config)
            input_embedding_tensor = torch.from_numpy(input_embedding_np).unsqueeze(0).to(torch.float32).to(device)

            # Get model predictions
            emotion_outputs, updated_memory_embedding = model(input_embedding_tensor)

            # Update memory
            memory.update(text_input)
            
            # Update model's memory embedding for next turn
            model.memory_embedding.data = updated_memory_embedding.data

            # Process emotion values
            emotion_state = {name: value.item() for name, value in emotion_outputs.items()}
            tracker.update(emotion_state)
            
            # Get smoothed state for more stable output
            smoothed_state = tracker.get_smoothed_state()

            # Display results
            print("\nYuna's Emotional State:")
            for emotion_name in emotion_names:
                raw_value = emotion_state[emotion_name]
                smooth_value = smoothed_state[emotion_name]
                print(f"{emotion_name}: {smooth_value:.4f} (raw: {raw_value:.4f})")
                
            print(f"\nMemory Summary: '{memory.get_recent(3)}'") 

if __name__ == "__main__":
    config = load_config()
    inference(config)