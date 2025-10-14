import json
import os
import re
import time
import urllib.parse
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import random

def get_config(config_path='static/config.json', config=None):
    default_config = {
        "ai": {
            "names": ["Yuki", "Yuna"],
            "bos": ["<|endoftext|>", True],
            "kokoro": False,
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
            "yuna_default_model": ["lib/models/yuna/yuna-ai-v4-miru-loli-mlx"],
            "voice_default_model": ["lib/models/agi/hanasu/config.json", "lib/models/agi/hanasu/G_46000.pth"],
            "device": "mps",
            "yuna_text_mode": "mlxvlm",
            "yuna_audio_mode": "hanasu",
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
    }

    if not os.path.exists(config_path): return default_config
    mode = 'r' if config is None else 'w'
    with open(config_path, mode) as f: return json.load(f) if config is None else json.dump(config, f, indent=4)

def clearText(text):
    text = text[0] if isinstance(text, tuple) else text
    original_text = text
    replacements = [('</yuki>', ''), ('</yuna>', ''), ('<yuki>', ''), ('<yuna>', '')]
    for old, new in replacements: text = text.replace(old, new)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+', '', text)
    text = re.sub(r':-?\)|:-?\(|;-?\)|:-?D|:-?P', '', text)
    text = ' '.join(text.split()).strip()
    if text != original_text: return clearText(text)
    return text

def calculate_similarity(embedding1, embedding2):
    if isinstance(embedding1, torch.Tensor): embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor): embedding2 = embedding2.cpu().numpy()
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1, embedding2)