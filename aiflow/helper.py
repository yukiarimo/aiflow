import json
import os
import re
import time
import urllib.parse
from contextlib import contextmanager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def get_config(config_path='static/config.json', config=None):
    default_config = {
        "ai": {
            "names": ["Yuki", "Yuna"],
            "himitsu": False,
            "agi": False,
            "emotions": False,
            "miru": False,
            "search": False,
            "audio": False,
            "mind": False,
            "voice": False,
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
            "use_mmap": False,
            "flash_attn": True,
            "use_mlock": True,
            "offload_kqv": True
        },
        "server": {
            "port": "",
            "url": "",
            "yuna_default_model": "yuna-ai-v4-q6_k",
            "miru_model_type": "moondream",
            "miru_default_model": "yuna-ai-miru-v0.gguf",
            "eyes_default_model": "yuna-ai-miru-eye-v0.gguf",
            "voice_default_model": "yuna-ai-voice-v1",
            "voice_model_config": ["YunaAi.ckpt", "YunaAi.pth"],
            "device": "mps",
            "yuna_text_mode": "koboldcpp",
            "yuna_audio_mode": "siri",
            "yuna_reference_audio": "static/audio/reference.wav"
        },
        "settings": {
            "pseudo_api": False,
            "fuctions": False,
            "notifications": False,
            "customConfig": True,
            "sounds": True,
            "use_history": True,
            "background_call": True,
            "nsfw_filter": False,
            "streaming": True,
            "default_history_file": "history_template:general.json",
            "default_kanojo": "Yuna",
            "default_prompt_template": "dialog"
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