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
    original_text = text
    replacements = [('</yuki>', ''), ('</yuna>', ''), ('<yuki>', ''), ('<yuna>', '')]
    for old, new in replacements: text = text.replace(old, new)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+', '', text)
    text = re.sub(r':-?\)|:-?\(|;-?\)|:-?D|:-?P', '', text)
    text = ' '.join(text.split()).strip()
    if text != original_text: return clearText(text)
    return text

def search_web(query, base_url='https://www.google.com', process_data=False):
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0'
    ]
    query = re.sub(r'[+\-\"\\/*^|<>~`]', '', query)
    query = re.sub(r'\s+', ' ', query).strip()[:300]
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.5', 'Referer': 'https://duckduckgo.com/', 'DNT': '1', 'Connection': 'keep-alive', 'Upgrade-Insecure-Requests': '1'}
    session = requests.Session()
    for attempt in range(3):
        try:
            response = session.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(random.uniform(2, 5))
            else:
                return []
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for result in soup.select('.result'):
        try:
            title_element = result.select_one('.result__a')
            title = title_element.get_text().strip() if title_element else ""
            url_element = result.select_one('.result__url')
            url = f"https://{url_element.get_text().strip()}" if url_element else ""
            link_element = result.select_one('a.result__a')
            if link_element and link_element.has_attr('href'):
                href = link_element['href']
                if href.startswith('http') and '/duckduckgo.com/' not in href:
                    url = href
            desc_element = result.select_one('.result__snippet')
            description = desc_element.get_text().strip() if desc_element else ""
            site = url_element.get_text().strip() if url_element else ""
            if title and (url or site):
                results.append({'title': title, 'description': description, 'url': url, 'site': site})
        except Exception:
            continue
    return results

def get_transcript(url): pass

def calculate_similarity(embedding1, embedding2):
    if isinstance(embedding1, torch.Tensor): embedding1 = embedding1.cpu().numpy()
    if isinstance(embedding2, torch.Tensor): embedding2 = embedding2.cpu().numpy()
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return np.dot(embedding1, embedding2)