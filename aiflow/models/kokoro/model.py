import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        x = self.proj(x)
        return x

class GatedResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.ff(x)
        x = self.norm2(x)
        
        gate = torch.sigmoid(self.gate(torch.cat([x, residual], dim=-1)))
        return gate * x + (1 - gate) * residual

class MemoryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = config["memory_embedding_size"]
        
        # Initial projection
        self.linear = nn.Linear(config["embedding_dimensions"], self.embedding_size)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)
        
        # Advanced processing
        self.norm = nn.LayerNorm(self.embedding_size)
        self.self_attention = SelfAttention(self.embedding_size)
        self.residual_blocks = nn.Sequential(
            GatedResidualBlock(self.embedding_size),
            GatedResidualBlock(self.embedding_size)
        )
        self.final_norm = nn.LayerNorm(self.embedding_size)
        
    def forward(self, memory_embedding):
        x = F.leaky_relu(self.linear(memory_embedding), negative_slope=0.01)
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        x = self.norm(x)
        x = x + self.self_attention(x)
        x = self.residual_blocks(x)
        x = self.final_norm(x)
        return x.squeeze(1)  # Remove sequence dimension

class InputModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = config["input_embedding_size"]
        
        # Initial projection
        self.linear = nn.Linear(config["embedding_dimensions"], self.embedding_size)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.0)
        
        # Advanced processing
        self.norm = nn.LayerNorm(self.embedding_size)
        self.residual_blocks = nn.Sequential(
            GatedResidualBlock(self.embedding_size),
            GatedResidualBlock(self.embedding_size)
        )
        self.final_norm = nn.LayerNorm(self.embedding_size)

    def forward(self, input_embedding):
        x = F.leaky_relu(self.linear(input_embedding), negative_slope=0.01)
        x = self.norm(x)
        x = self.residual_blocks(x)
        x = self.final_norm(x)
        return x

class EmotionMixingLayer(nn.Module):
    def __init__(self, num_emotions, hidden_size):
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_size = hidden_size
        
        # Emotion mixing matrix (each emotion affects others)
        self.mixing_weights = nn.Parameter(torch.zeros(num_emotions, num_emotions))
        self.mixing_bias = nn.Parameter(torch.zeros(num_emotions))
        
        # Initialize with identity-ish matrix with small random variation
        nn.init.eye_(self.mixing_weights)
        self.mixing_weights.data += torch.randn_like(self.mixing_weights.data) * 0.01
        
    def forward(self, emotion_values):
        # emotion_values: [batch_size, num_emotions]
        batch_size = emotion_values.shape[0]
        mixed_emotions = torch.matmul(emotion_values, self.mixing_weights) + self.mixing_bias
        return torch.tanh(mixed_emotions)  # Apply tanh to keep values in range

class StateModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["state_hidden_size"]
        self.num_emotions = config["num_emotions"]
        self.memory_embedding_size = config["memory_embedding_size"]
        self.emotion_names = config["emotion_names"]
        
        # Initial emotion processors - shared layers first
        self.shared_layer = nn.Sequential(
            nn.Linear(self.memory_embedding_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        
        # Individual emotion networks with residual connections
        self.emotion_processors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
                nn.LayerNorm(self.hidden_size // 4),
                nn.LeakyReLU(0.01),
                nn.Linear(self.hidden_size // 4, 1)
            ) for name in self.emotion_names
        })
        
        # Emotion mixing for interactions between different emotions
        self.emotion_mixing = EmotionMixingLayer(len(self.emotion_names), self.hidden_size)

    def forward(self, memory_embedding):
        # Process memory with shared layers
        shared_features = self.shared_layer(memory_embedding)
        
        # Process each emotion individually
        emotion_values_dict = {}
        emotion_values_list = []
        
        for name in self.emotion_names:
            value = self.emotion_processors[name](shared_features)
            emotion_values_dict[name] = value
            emotion_values_list.append(value)
            
        # Combine all emotion values for mixing
        emotion_values_cat = torch.cat(emotion_values_list, dim=1)  # [batch_size, num_emotions]
        
        # Mix emotions (allow them to influence each other)
        mixed_emotions = self.emotion_mixing(emotion_values_cat)
        
        # Repackage as dictionary
        result = {}
        for i, name in enumerate(self.emotion_names):
            result[name] = mixed_emotions[:, i:i+1]
            
        return result

class GatedMemoryUpdate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.update_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.reset_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.memory_candidate = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, memory, input_embedding):
        combined = torch.cat([memory, input_embedding], dim=1)
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        
        reset_memory = reset_gate * memory
        memory_candidate_input = torch.cat([reset_memory, input_embedding], dim=1)
        memory_candidate = torch.tanh(self.memory_candidate(memory_candidate_input))
        
        new_memory = (1 - update_gate) * memory + update_gate * memory_candidate
        return new_memory

class EmotionalModel(nn.Module):
    def __init__(self, config):
        # Convert model to FP16 during initialization
        super().__init__()
        self.config = config

        self.memory_module = MemoryModule(config)
        self.input_module = InputModule(config)
        self.state_module = StateModule(config)
        self.history_weight = config["history_weight"]
        
        # Advanced memory update mechanism
        self.gated_memory_update = GatedMemoryUpdate(config["embedding_dimensions"])

        # Initialize memory embedding
        self.memory_embedding = nn.Parameter(torch.randn(1, config["embedding_dimensions"]))
        
    def forward(self, input_embedding_raw):
        batch_size = input_embedding_raw.shape[0]
        
        # 1. Process Input
        input_embedding = self.input_module(input_embedding_raw)
        
        # 2. Process Memory (using the *current* memory embedding)
        # Expand memory to match batch size
        expanded_memory = self.memory_embedding.expand(batch_size, -1)
        memory_processed = self.memory_module(expanded_memory)
        
        # 3. Update memory with sophisticated gating mechanism - only use the first example
        input_mean = input_embedding.mean(dim=0, keepdim=True)
        self.memory_embedding.data = self.gated_memory_update(
            self.memory_embedding.data, 
            input_mean.data
        )
        
        # 4. State Prediction from processed Memory
        emotion_values = self.state_module(memory_processed)

        return emotion_values, self.memory_embedding