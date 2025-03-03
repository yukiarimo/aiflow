import llama_cpp
import numpy as np
import json
import os
import hashlib
from functools import lru_cache
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import EmotionalModel

# Global variable to hold the loaded LLaMA model
llm_instance = None
llm_model_path = None # To track if model path changed

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_llm_model(config):
    global llm_instance, llm_model_path
    model_path = config["embedding_model_path"]
    if llm_instance is None or model_path != llm_model_path:
        print(f"Loading LLaMA model from: {model_path}")
        llm_instance = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            use_mmap=False,
            verbose=False,
            n_gpu_layers=-1
        )
        llm_model_path = model_path
    return llm_instance

# Create a hash of text for caching
def text_to_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# LRU cache for embeddings in memory
@lru_cache(maxsize=1000)
def _get_cached_embedding(text_hash):
    cache_dir = "embedding_cache"
    cache_path = os.path.join(cache_dir, f"{text_hash}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None

def get_text_embedding(text, config, use_cache=True):
    """Get text embedding with disk caching for efficiency"""
    global llm_instance
    
    # Setup cache
    cache_dir = "embedding_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the text for cache key
    text_hash = text_to_hash(text)
    cache_path = os.path.join(cache_dir, f"{text_hash}.npy")
    
    # Try to get from cache first if enabled
    if use_cache:
        cached_embedding = _get_cached_embedding(text_hash)
        if cached_embedding is not None:
            return cached_embedding
    
    # If not in cache, compute embedding
    if llm_instance is None:
        llm = get_llm_model(config)
    else:
        llm = llm_instance

    # Handle potential embedding failures with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            embeddings = llm.create_embedding(text)
            all_embeddings = [np.array(emb_data['embedding']) for emb_data in embeddings['data']]
            
            if not all_embeddings:
                print(f"Warning: No embeddings returned for '{text[:20]}...', using zero vector.")
                averaged_embedding = np.zeros(config["embedding_dimensions"], dtype=np.float32)
            else:
                stacked_embeddings = np.vstack(all_embeddings)
                averaged_embedding = np.mean(stacked_embeddings, axis=0)
                
            # Save to cache if using cache
            if use_cache:
                np.save(cache_path, averaged_embedding)
                
            return averaged_embedding.astype(np.float32)
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Embedding failed: {str(e)}. Retrying ({attempt+1}/{max_retries})...")
            else:
                print(f"Embedding failed after {max_retries} attempts. Using zero vector.")
                averaged_embedding = np.zeros(config["embedding_dimensions"], dtype=np.float32)
                return averaged_embedding

class EmotionalDataset(Dataset):
    def __init__(self, data_path, config, cache_embeddings=True):
        self.config = config
        self.data = self._load_data(data_path)
        self.emotion_names = config["emotion_names"]
        self.cache_dir = os.path.join(os.path.dirname(data_path), "embedding_cache")
        self.cache_embeddings = cache_embeddings
        
        # Create cache directory if needed
        if self.cache_embeddings:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Pre-compute embeddings if caching is enabled
        if self.cache_embeddings:
            self._precompute_embeddings()

    def _load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def _precompute_embeddings(self):
        """Precompute embeddings one at a time - slower but more reliable"""
        pending_items = []
        for idx, item in enumerate(self.data):
            memory_cache = os.path.join(self.cache_dir, f"memory_{idx}.npy")
            input_cache = os.path.join(self.cache_dir, f"input_{idx}.npy")
            if not (os.path.exists(memory_cache) and os.path.exists(input_cache)):
                pending_items.append((idx, item))
        
        if not pending_items:
            print("All embeddings already cached. Skipping precomputation.")
            return
        
        print(f"Precomputing embeddings for {len(pending_items)} items...")
        
        # Process each item sequentially
        for idx, item in tqdm(pending_items, desc="Caching embeddings"):
            memory_cache = os.path.join(self.cache_dir, f"memory_{idx}.npy")
            input_cache = os.path.join(self.cache_dir, f"input_{idx}.npy")
            
            if not os.path.exists(memory_cache):
                memory_embedding = get_text_embedding(item["memory"], self.config)
                np.save(memory_cache, memory_embedding)
            
            if not os.path.exists(input_cache):
                input_embedding = get_text_embedding(item["input"], self.config)
                np.save(input_cache, input_embedding)

    def _get_embedding(self, text, idx, prefix):
        """Get embedding from cache if available, otherwise compute it"""
        cache_path = os.path.join(self.cache_dir, f"{prefix}_{idx}.npy")
        
        if self.cache_embeddings and os.path.exists(cache_path):
            return np.load(cache_path)
        else:
            embedding = get_text_embedding(text, self.config)
            if self.cache_embeddings:
                np.save(cache_path, embedding)
            return embedding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        memory_text = item["memory"]
        input_text = item["input"]
        target_state = item["state"]

        memory_embedding = self._get_embedding(memory_text, idx, "memory")
        input_embedding = self._get_embedding(input_text, idx, "input")

        # Return target as a flat vector without the extra dimension
        target_vector = torch.tensor([float(target_state[emotion]) for emotion in self.emotion_names], dtype=torch.float32)

        return torch.from_numpy(memory_embedding), torch.from_numpy(input_embedding), target_vector

def _process_embedding_item(args):
    """Process a single embedding item - must be at module level for multiprocessing"""
    idx, item, config, cache_dir = args
    memory_cache = os.path.join(cache_dir, f"memory_{idx}.npy")
    input_cache = os.path.join(cache_dir, f"input_{idx}.npy")
    
    if not os.path.exists(memory_cache):
        memory_embedding = get_text_embedding(item["memory"], config)
        np.save(memory_cache, memory_embedding)
    
    if not os.path.exists(input_cache):
        input_embedding = get_text_embedding(item["input"], config)
        np.save(input_cache, input_embedding)
    
    return idx

def create_dataloader(data_path, config, shuffle=True, cache_embeddings=True, batch_size=None):
    dataset = EmotionalDataset(data_path, config, cache_embeddings=cache_embeddings)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size if batch_size is not None else config["batch_size"], 
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for macOS compatibility
    )
    return dataloader

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_param_count(count):
    """Format parameter count in human-readable form (K, M, B)"""
    if count < 1_000:
        return f"{count}"
    elif count < 1_000_000:
        return f"{count/1_000:.2f}K"
    elif count < 1_000_000_000:
        return f"{count/1_000_000:.2f}M"
    else:
        return f"{count/1_000_000_000:.2f}B"

def analyze_model():
    # Load configuration
    config = load_config()
    
    # Initialize the model
    model = EmotionalModel(config)
    
    # Count parameters
    param_count = count_parameters(model)
    
    # Print model details
    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {param_count:,} ({format_param_count(param_count)})")
    
    # Analyze by layer groups
    param_groups = {}
    
    # Count parameters by module
    param_groups["Memory Module"] = sum(p.numel() for p in model.memory_module.parameters() if p.requires_grad)
    param_groups["Input Module"] = sum(p.numel() for p in model.input_module.parameters() if p.requires_grad)
    param_groups["State Module"] = sum(p.numel() for p in model.state_module.parameters() if p.requires_grad)
    param_groups["Memory Update"] = sum(p.numel() for p in model.gated_memory_update.parameters() if p.requires_grad)
    param_groups["Memory Embedding"] = model.memory_embedding.numel()
    
    # Print module parameters
    print("\nParameters by module:")
    for name, count in param_groups.items():
        pct = 100.0 * count / param_count
        print(f"  {name}: {count:,} ({format_param_count(count)}) - {pct:.1f}%")
    
    # Print layer dimensions
    print("\nKey dimensions:")
    print(f"  Memory embedding size: {config['memory_embedding_size']}")
    print(f"  Input embedding size: {config['input_embedding_size']}")
    print(f"  State hidden size: {config['state_hidden_size']}")
    print(f"  Number of emotions: {config['num_emotions']}")
    
    # If a saved model exists, try to load and analyze it
    try:
        saved_path = "emotional_model_final.pth"
        if not torch.cuda.is_available():
            state_dict = torch.load(saved_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(saved_path)
        
        print(f"\nLoaded saved model from: {saved_path}")
    except:
        try:
            saved_path = "checkpoints/best_model.pth"
            if not torch.cuda.is_available():
                checkpoint = torch.load(saved_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(saved_path)
            
            print(f"\nLoaded checkpoint from: {saved_path}")
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
            print(f"  Training loss: {checkpoint['train_loss']:.4f}")
            print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
        except:
            print("\nNo saved model found.")