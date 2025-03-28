import json
import os
import re
import time
import urllib.parse
import llama_cpp
import numpy as np
from tqdm import tqdm
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from functools import lru_cache
from aiflow.models.kokoro import model as kokoro

import torch
import os
from aiflow.models.kokorox import KokoroXProcessor
from aiflow.utils import load_config, get_text_embedding, load_kokoro_model

def get_config(config_path='static/config.json', config=None):
    default_config = {
        "ai": {
            "names": ["Yuki", "Yuna"],
            "himitsu": False,
            "agi": False,
            "emotions": False,
            "miru": False,
            "audio": False,
            "mind": False,
            "hanasu": False,
            "max_new_tokens": 1024,
            "context_length": 16384,
            "temperature": 0.7,
            "repetition_penalty": 1.11,
            "last_n_tokens_size": 128,
            "seed": -1,
            "top_k": 100,
            "top_p": 1,
            "stop": ["<yuki>", "</yuki>", "<yuna>", "</yuna>", "<hito>", "</hito>", "<data>", "</data>", "<kanojo>", "</kanojo>"],
            "batch_size": 2048,
            "threads": 8,
            "gpu_layers": -1,
            "use_mmap": True,
            "flash_attn": True,
            "use_mlock": True,
            "offload_kqv": True
        },
        "server": {
            "url": "",
            "yuna_default_model": "lib/models/yuna/yuna-ai-v4-q5_k_m.gguf",
            "miru_default_model": ["yuna-ai-miru-v0.gguf", "yuna-ai-miru-eye-v0.gguf"],
            "voice_default_model": "yuna-ai-voice-v1",
            "voice_model_config": ["YunaAi.ckpt", "YunaAi.pth"],
            "device": "mps",
            "yuna_text_mode": "koboldcpp",
            "yuna_miru_mode": "native",
            "yuna_audio_mode": "siri",
            "yuna_reference_audio": "static/audio/reference.wav"
        },
        "settings": {
            "fuctions": True,
            "use_history": True,
            "customConfig": True,
            "sounds": True,
            "background_call": True,
            "streaming": True,
            "default_history_file": "history_template:general.json",
            "default_kanojo": "Yuna"
        },
        "security": {
            "secret_key": "YourSecretKeyHere123!",
            "encryption_key": "zWZnu-lxHCTgY_EqlH4raJjxNJIgPlvXFbdk45bca_I=",
            "11labs_key": "Your11LabsKeyHere123!"
        }
    }

    if not os.path.exists(config_path):
        return default_config

    mode = 'r' if config is None else 'w'
    with open(config_path, mode) as f:
        return json.load(f) if config is None else json.dump(config, f, indent=4)

def clearText(text):
    text = text.replace('</yuki>', '').replace('</yuna>', '').replace('<yuki>', '').replace('<yuna>', '')
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+', '', text)
    text = re.sub(r':-?\)|:-?\(|;-?\)|:-?D|:-?P', '', text)
    return ' '.join(text.split()).strip()

@contextmanager
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        yield driver
    finally:
        driver.quit()

def search_web(search_query, base_url='https://www.google.com', process_data=False):
    encoded_query = urllib.parse.quote(search_query)
    search_url = f'{base_url}/search?q={encoded_query}'

    with get_driver() as driver:
        try:
            driver.get(search_url)
        except Exception as e:
            print(f"Error navigating to {search_url}: {e}")
            return None, None, None

        answer = driver.execute_script("""
            var ans = document.querySelector('.kno-rdesc > span') || document.querySelector('.hgKElc');
            return ans ? ans.textContent.trim() : 'Answer not found.';
        """)

        search_results = driver.execute_script("""
            return Array.from(document.querySelectorAll('.g')).map(result => {
                const link = result.querySelector('.yuRUbf a')?.href || '';
                const title = result.querySelector('.yuRUbf a h3')?.textContent.trim() || '';
                const description = result.querySelector('.VwiC3b')?.textContent.trim() || '';
                return { 'Link': link, 'Title': title, 'Description': description };
            }).filter(r => r.Link && r.Title && r.Description);
        """)

    image_urls = search_images(search_query, base_url)
    return answer, search_results, image_urls

def search_images(search_query, base_url='https://www.google.com'):
    encoded_query = urllib.parse.quote(search_query)
    image_search_url = f'{base_url}/search?q={encoded_query}&tbm=isch'

    with get_driver() as driver:
        try:
            driver.get(image_search_url)
        except Exception as e:
            print(f"Error navigating to {image_search_url}: {e}")
            return None

        image_urls = driver.execute_script("""
            let images = Array.from(document.querySelectorAll('img'));
            return images.slice(0, 3).map(img => img.src);
        """)
    return image_urls

def get_html(url):
    with get_driver() as driver:
        try:
            driver.get(url)
            return driver.page_source
        except Exception as e:
            print(f"Error navigating to {url}: {e}")
            return None

def get_transcript(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    with webdriver.Chrome(service=service, options=chrome_options) as driver:
        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            time.sleep(2)

            driver.execute_script("""
                if (window.trustedTypes && window.trustedTypes.createPolicy && !window.trustedTypes.defaultPolicy) {
                    window.trustedTypes.createPolicy('default', {
                        createHTML: string => string
                    });
                }
            """)

            transcript = driver.execute_script("""
                // JavaScript to extract YouTube transcript
                const button = document.querySelector('button.ytp-transcript');
                if (button) {
                    button.click();
                    const texts = Array.from(document.querySelectorAll('.cue-group .cue')).map(el => el.textContent.trim());
                    return texts.join(' ');
                } else {
                    return 'Transcript not available.';
                }
            """)
            return transcript

        except Exception as e:
            print(f"Error getting transcript from {url}: {e}")
            return None

"""

BELOW IS FOR CUSTOM MODELS

"""

# Global variable to hold the loaded LLaMA model
llm_instance = None
llm_model_path = None  # To track if model path changed

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_llm_model(config):
    """Get or initialize the LLaMA model for embeddings."""
    global llm_instance, llm_model_path
    model_path = config["embedding_model_path"]

    if llm_instance is None or model_path != llm_model_path:
        print(f"Loading embedding model from: {model_path}")
        llm_instance = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            use_mmap=False,
            verbose=False,
            n_gpu_layers=-1
        )
        llm_model_path = model_path

    return llm_instance

def text_to_hash(text):
    """Create a hash of text for caching."""
    return hashlib.md5(text.encode()).hexdigest()

@lru_cache(maxsize=2000)
def _get_cached_embedding(text_hash):
    """Retrieve cached embedding by hash."""
    cache_dir = "embedding_cache"
    cache_path = os.path.join(cache_dir, f"{text_hash}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None

def get_text_embedding(text, config=None):
    """Get text embedding with disk caching for efficiency."""
    global llm_instance

    # Load config if not provided
    if config is None:
        config = load_config()

    # Setup cache
    cache_dir = "embedding_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Create a hash of the text for cache key
    text_hash = text_to_hash(text)
    cache_path = os.path.join(cache_dir, f"{text_hash}.npy")

    # Try to get from cache first
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

            # Save to cache
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

def load_kokoro_model(config):
    """Load the Kokoro emotional model."""
    device = torch.device(config["device"])

    # Initialize Kokoro model
    kokoro_model = kokoro.EmotionalModel(config).to(device)

    # Load checkpoint
    if os.path.exists(config["emotion_model_path"]):
        checkpoint = torch.load(config["emotion_model_path"], map_location=device)
        if 'model_state_dict' in checkpoint:
            kokoro_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            kokoro_model.load_state_dict(checkpoint)

        print(f"Loaded Kokoro model from {config['emotion_model_path']}")
    else:
        print(f"Warning: No Kokoro model found at {config['emotion_model_path']}, using untrained model")

    kokoro_model.eval()
    return kokoro_model

def count_tokens(text, tokenizer=None):
    """Count the number of tokens in text using a tokenizer or simple word counting."""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Simple word-based tokenization
        return len(text.split())

def chunk_text(text, max_tokens=512, overlap=50):
    """Split text into chunks with optional overlap."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_tokens - overlap):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)

    return chunks

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    if isinstance(embedding1, torch.Tensor):
        embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor):
        embedding2 = embedding2.cpu().numpy()

    # Normalize embeddings
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    return similarity