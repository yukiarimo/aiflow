import torch
import torch.nn as nn
import numpy as np
import mlx.core as mx
from aiflow.models.yuna_vlm.utils import load


class EmbeddingGenerator:
	def __init__(self, config):
		model_path = config.get("embedding_model_path")
		if not model_path:
			raise ValueError("Config is missing 'embedding_model_path'")
		print(f"Loading embedding model from: {model_path}...")
		self.model, self.processor = load(model_path)
		print("Embedding model loaded successfully.")

	def get_embedding(self, text):
		inputs = self.processor(text=[text], images=None, return_tensors="pt", padding=True)
		input_ids = mx.array(np.array(inputs['input_ids']))

		text_embeddings, _ = self.model.get_input_embeddings(input_ids=input_ids, pixel_values=None)

		# Mean pooling and normalization
		averaged_embedding = mx.mean(text_embeddings, axis=1).squeeze().astype(mx.float32)
		norm = mx.linalg.norm(averaged_embedding) + 1e-8
		normalized_embedding = averaged_embedding / norm

		return np.array(normalized_embedding, dtype=np.float32)


class MemoryModule(nn.Module):
	def __init__(self, config):
		super().__init__()
		dim = config["embedding_dimensions"]
		hidden = config["memory_hidden_size"]
		self.network = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(config["dropout_rate"]), nn.Linear(hidden, hidden))

	def forward(self, x):
		return self.network(x)


class StateModule(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.emotion_names = config["emotion_names"]
		mem_size = config["memory_hidden_size"]
		state_size = config["state_hidden_size"]

		# Predicts all 6 emotions at once (Massive optimization over ModuleDict)
		self.network = nn.Sequential(nn.Linear(mem_size, state_size), nn.LayerNorm(state_size), nn.GELU(), nn.Dropout(config["dropout_rate"]), nn.Linear(state_size, len(self.emotion_names)), nn.Tanh())  # Output bounded to [-1, 1]

	def forward(self, processed_memory):
		return self.network(processed_memory)


class GatedMemoryUpdate(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		self.update_gate = nn.Linear(embed_dim * 2, embed_dim)
		self.reset_gate = nn.Linear(embed_dim * 2, embed_dim)
		self.candidate = nn.Linear(embed_dim * 2, embed_dim)

	def forward(self, memory_embedding, input_embedding):
		combined = torch.cat([memory_embedding, input_embedding], dim=1)

		z = torch.sigmoid(self.update_gate(combined))
		r = torch.sigmoid(self.reset_gate(combined))

		candidate_input = torch.cat([r * memory_embedding, input_embedding], dim=1)
		h_tilde = torch.tanh(self.candidate(candidate_input))

		new_memory = (1 - z) * memory_embedding + z * h_tilde
		return new_memory


class KokoroModel(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.memory_module = MemoryModule(config)
		self.state_module = StateModule(config)
		self.gated_memory_update = GatedMemoryUpdate(config["embedding_dimensions"])
		self.emotion_names = config["emotion_names"]

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	def forward(self, input_embedding, memory_embedding):
		processed_memory = self.memory_module(memory_embedding)
		emotion_tensor = self.state_module(processed_memory)
		new_memory = self.gated_memory_update(memory_embedding, input_embedding)

		return emotion_tensor, new_memory
