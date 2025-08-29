import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
from mlx_vlm import load

class EmbeddingGenerator:
    def __init__(self, config):
        model_path = config.get("embedding_model_path")
        if not model_path: raise ValueError("Config is missing 'embedding_model_path'")
        print(f"Loading embedding model from: {model_path}...")
        self.model, self.processor = load(model_path)
        print("Embedding model loaded successfully.")

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], images=None, return_tensors="pt", padding=True)
        numpy_ids = np.array(inputs['input_ids'])
        input_ids = mx.array(numpy_ids)
        text_embeddings, _ = self.model.get_input_embeddings(input_ids=input_ids, pixel_values=None)
        averaged_embedding = mx.mean(text_embeddings, axis=1).squeeze()
        float32_embedding = averaged_embedding.astype(mx.float32)
        norm = mx.linalg.norm(float32_embedding) + 1e-8
        normalized_embedding = float32_embedding / norm
        return np.array(normalized_embedding, dtype=np.float32)

class MemoryModule(nn.Module):
    """Processes the raw memory embedding into a rich representation for the StateModule."""
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config["embedding_dimensions"], config["memory_hidden_size"]),
            nn.LayerNorm(config["memory_hidden_size"]),
            nn.GELU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(config["memory_hidden_size"], config["memory_hidden_size"])
        )

    def forward(self, x): return self.network(x)

class StateModule(nn.Module):
    """
    Predicts the emotional state based on the processed memory. This contains six independent regression heads, one for each emotion.
    """
    def __init__(self, config):
        super().__init__()
        self.emotion_names = config["emotion_names"]
        memory_size = config["memory_hidden_size"]
        state_size = config["state_hidden_size"]

        self.emotion_processors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(memory_size, state_size),
                nn.LayerNorm(state_size),
                nn.GELU(),
                nn.Dropout(config["dropout_rate"]),
                nn.Linear(state_size, 1)
            ) for name in self.emotion_names
        })
        self.output_activation = nn.Tanh() # Output is always in [-1, 1]

    def forward(self, processed_memory):
        emotion_activations = [self.emotion_processors[name](processed_memory) for name in self.emotion_names]
        emotion_vector = torch.cat(emotion_activations, dim=1)
        out = self.output_activation(emotion_vector)
        return {name: out[:, i] for i, name in enumerate(self.emotion_names)}

class GatedMemoryUpdate(nn.Module):
    """Updates the raw memory embedding for the next turn."""
    def __init__(self, embed_dim):
        super().__init__()
        self.update_gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.reset_gate_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.candidate_linear = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, memory_embedding, input_embedding):
        combined = torch.cat([memory_embedding, input_embedding], dim=1)
        update_gate = torch.sigmoid(self.update_gate_linear(combined))
        reset_gate = torch.sigmoid(self.reset_gate_linear(combined))
        candidate_input = torch.cat([reset_gate * memory_embedding, input_embedding], dim=1)
        candidate_memory = torch.tanh(self.candidate_linear(candidate_input))
        new_memory = (1 - update_gate) * memory_embedding + update_gate * candidate_memory
        return new_memory

class KokoroModel(nn.Module):
    """
    The Kokoro model from the paper.
    """
    def __init__(self, config):
        super().__init__()
        self.memory_module = MemoryModule(config)
        self.state_module = StateModule(config)
        self.gated_memory_update = GatedMemoryUpdate(config["embedding_dimensions"])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_embedding, memory_embedding):
        processed_memory = self.memory_module(memory_embedding)
        emotion_predictions = self.state_module(processed_memory)
        new_memory_embedding = self.gated_memory_update(memory_embedding, input_embedding)
        return emotion_predictions, new_memory_embedding