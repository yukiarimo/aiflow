import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from uta import MusicTokenizer

class GPTConfig:
    def __init__(self, vocab_size, block_size):
        # Model hyperparameters
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = 384 #384
        self.n_head = 6
        self.n_layer = 6
        self.dropout = 0.01 #0.2
        self.batch_size = 32 #64
        self.learning_rate = 4e-5 #3e-4
        self.max_iters = 5000 #5000
        self.eval_interval = 200 #200
        self.eval_iters = 20 #200
        self.device = 'mps'

class DataLoader:
    def __init__(self, data, block_size, batch_size, device):
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split):
        data = self.data[split]
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, idx, targets=None):
        B, T = idx.size()
        token_embeddings = self.token_embedding(idx)  # [batch_size, seq_len, n_embd]
        
        position_embeddings = self.position_embedding[:, :T, :]  # [1, seq_len, n_embd]
        x = token_embeddings + position_embeddings
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate the loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]  # Use self.config.block_size here
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_embd = n_embd // n_head
        self.qkv = nn.Linear(n_embd, n_embd * 3)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.n_embd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.n_embd ** 0.5))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Trainer:
    def __init__(self, model, data_loader, tokenizer, config):
        self.model = model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def train(self):
        self.model.train()
        for iter in range(self.config.max_iters):
            xb, yb = self.data_loader.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                # Save the model checkpoint
                torch.save(self.model.state_dict(), f'checkpoint_{iter}.pth')

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.data_loader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    @torch.no_grad()
    def generate(self, initial_tokens, max_new_tokens):
        # Convert the initial_tokens list to a tensor and add a batch dimension
        context = torch.tensor(initial_tokens, dtype=torch.long, device=self.config.device).unsqueeze(0)
        generated_tokens = self.model.generate(context, max_new_tokens)  # Call the generate method of the GPT class
        return generated_tokens
    
    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name):
        if os.path.isfile(file_name):
            self.model.load_state_dict(torch.load(file_name))
            print(f"Model loaded from {file_name}")
        else:
            print(f"No saved model found at {file_name}")

# Create a custom tokenizer for the GPT model that uses the MusicTokenizer's vocabulary
class GPTMusicTokenizer:
    def __init__(self, music_tokenizer):
        self.music_tokenizer = music_tokenizer
    
    @property
    def vocab_size(self):
        return len(self.music_tokenizer.token_to_idx)
    
    def encode(self, notes):
        return [self.music_tokenizer.tokenize(note) for note in notes]
    
    def decode(self, tokens):
        return [self.music_tokenizer.detokenize(token) for token in tokens]