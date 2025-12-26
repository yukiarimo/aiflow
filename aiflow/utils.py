import json
import os
import re
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_config(config_path='static/config.json', config=None):
    default_config = {
        "yuna": {
            "audio": True,
            "batch_size": 512,
            "bos": [
                "<|endoftext|>",
                True
            ],
            "context_length": 16384,
            "flash_attn": True,
            "gpu_layers": -1,
            "hanasu": True,
            "kokoro": False,
            "last_n_tokens_size": 128,
            "max_new_tokens": 1024,
            "mind": True,
            "offload_kqv": True,
            "repetition_penalty": 1.1,
            "seed": -1,
            "stop": [
                "<yuki>",
                "</yuki>",
                "<yuna>",
                "</yuna>",
                "<hito>",
                "</hito>",
                "<data>",
                "</data>",
                "<kanojo>",
                "</kanojo>"
            ],
            "temperature": 0.7,
            "threads": 8,
            "top_k": 40,
            "top_p": 0.9,
            "use_mlock": True,
            "use_mmap": True
        },
        "server": {
            "kagi_key": "",
            "url": "",
            "voice_default_model": [
                "/Users/yuki/Documents/Github/aiflow/aiflow/models/hanasu/config.json",
                "/Users/yuki/Downloads/G_4000.pth"
            ],
            "yuna_audio_mode": "hanasu",
            "yuna_default_model": [
                "/Users/yuki/Documents/Github/mlx-vlm-main 2/looooolilast3"
            ],
            "yuna_text_mode": "mlxvlm",
            "sounds": True,
            "use_history": True
        }
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

def reverse_image_search(image_path, session_token=None):
    """
    Perform reverse image search using Kagi API

    Args:
        image_path: Path to the local image file
        session_token: Optional Kagi session token for authentication
    """
    url = "https://kagi.com/reverse/upload"

    try:
        # Open and read the image file
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}

            # Add session cookies if provided
            cookies = {}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Referer': 'https://kagi.com/images'
            }

            if session_token:
                cookies['kagi_session'] = session_token

            # Send POST request with the image
            response = requests.post(
                url,
                files=files,
                cookies=cookies,
                headers=headers,
                allow_redirects=True
            )

            print(f"Status Code: {response.status_code}")
            print(f"Final URL: {response.url}")

            # Check if we got redirected to login or signin
            if 'login' in response.url or 'signin' in response.url:
                print("\n❌ Error: Authentication failed.")
                print("The session token may be expired or invalid.")
                return

            # Check if we got a successful redirect to image results
            if '/images?' in response.url and 'reverse=upload' in response.url:
                print("\n✓ Image uploaded successfully!")
                print(f"🔗 View results in browser: {response.url}")

                # Parse the HTML to extract image results
                soup = BeautifulSoup(response.content, 'html.parser')

                # Try to find image results
                print("\n📸 Similar images found on the page:")

                # Look for the page title
                title = soup.find('title')
                if title:
                    print(f"Page title: {title.text}")

                image_results = soup.find_all('div', class_='_0_image_item')

                if image_results:
                    print(f"\nFound {len(image_results)} similar images:")
                    for i, item in enumerate(image_results, 1):
                        img_tag = item.find('img', class_='_0_img_src')
                        if not img_tag:
                            continue

                        title = item.get('data-title', 'N/A')
                        # The src is a relative proxy URL, prepend the domain
                        proxy_img_url = "https://kagi.com" + img_tag.get('src', '')
                        source_url = item.get('data-host_url', 'N/A')
                        width = item.get('data-width', 'N/A')
                        height = item.get('data-height', 'N/A')

                        print(f"\n--- Image {i} ---")
                        print(f"  Title: {title}")
                        print(f"  Dimensions: {width}x{height}")
                        print(f"  Source URL: {source_url}")
                        print(f"  Image Proxy URL: {proxy_img_url}")

                else:
                    print("\nCould not find image items on the page.")

                print("\nNote: Open the URL above in your browser to see the visual results.")
            else:
                print("\n⚠️  Unexpected response:")
                print(response.text[:1000])

    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error making request: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

class WebParser:
    """Safari Reader Mode-style article extractor with minimal dependencies."""

    def __init__(self):
        self.title = ''
        self.content = ''

    def parse(self, url='', html='', output='markdown', timeout=15):
        """
        Extract article content from URL or HTML.

        Args:
            url: Article URL to fetch
            html: Pre-fetched HTML content
            output: Output format ('markdown' or 'html')
            timeout: Request timeout in seconds

        Returns:
            Tuple of (title, content)
        """
        if not html and url:
            html = self._download(url, timeout)

        if not html:
            return '', ''

        # Clean HTML
        html = self._clean_html(html)

        # Extract metadata
        meta = self._extract_metadata(html)
        title = meta.get('title', '')

        # Find main content
        content_html = self._extract_content(html)

        if not content_html:
            return title, ''

        # Extract title from content if not found in metadata
        if not title:
            title = self._extract_title_from_content(content_html)

        # Convert to requested format
        if output == 'markdown':
            content = self._html_to_markdown(content_html, url)
        else:
            content = content_html

        # Clean up content
        text_only = self._strip_tags(content)
        if len(text_only.strip()) < 200:
            return '', ''

        return title.strip(), content.strip()

    def _download(self, url, timeout: int):
        """Download HTML from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()

            # Handle encoding
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding

            return response.text
        except Exception:
            return ''

    def _clean_html(self, html):
        """Remove unwanted elements from HTML."""
        # Remove comments
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<noscript[^>]*>.*?</noscript>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove hidden elements
        html = re.sub(r'<[^>]+style\s*=\s*["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>.*?</[^>]+>',
                     '', html, flags=re.DOTALL | re.IGNORECASE)

        return html

    def _extract_metadata(self, html):
        """Extract metadata from HTML."""
        meta = {}

        # Title from <title> tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1)
            title = re.sub(r'<[^>]+>', '', title)
            title = title.split('|')[0].split('-')[0].strip()
            meta['title'] = self._decode_html_entities(title)

        return meta

    def _extract_content(self, html):
        """Extract main article content using text density analysis."""
        # Find body content
        body_match = re.search(r'<body[^>]*>(.*?)</body>', html, re.IGNORECASE | re.DOTALL)
        if not body_match:
            return ''

        body_html = body_match.group(1)

        # Parse into blocks
        blocks = self._split_into_blocks(body_html)

        # Score each block
        best_block = None
        best_score = 0

        for block in blocks:
            score = self._score_block(block)
            if score > best_score:
                best_score = score
                best_block = block

        if best_block:
            return self._clean_content_block(best_block)

        return ''

    def _split_into_blocks(self, html) -> list:
        """Split HTML into content blocks."""
        # Find major container tags
        containers = re.findall(
            r'<(div|article|section|main)[^>]*>(.*?)</\1>',
            html,
            re.IGNORECASE | re.DOTALL
        )

        blocks = [match[1] for match in containers]

        if not blocks:
            blocks = [html]

        return blocks

    def _score_block(self, block) -> float:
        """Score a content block based on text density and content signals."""
        # Remove tags for text analysis
        text = self._strip_tags(block)
        text_len = len(text.strip())

        if text_len < 100:
            return 0

        # Count elements
        p_count = len(re.findall(r'<p[^>]*>', block, re.IGNORECASE))
        link_count = len(re.findall(r'<a[^>]*>', block, re.IGNORECASE))
        img_count = len(re.findall(r'<img[^>]*>', block, re.IGNORECASE))

        # Calculate score
        score = text_len
        score += p_count * 50  # Paragraphs are good
        score -= link_count * 20  # Too many links = navigation
        score += min(img_count * 30, 150)  # Images are good but cap it

        # Check for article keywords
        if re.search(r'(article|post|entry|content|main|story)', block, re.IGNORECASE):
            score += 100

        # Penalize navigation patterns
        if re.search(r'(nav|menu|sidebar|footer|header|comment)', block, re.IGNORECASE):
            score -= 200

        return score

    def _clean_content_block(self, block):
        """Clean extracted content block."""
        # Remove navigation elements
        block = re.sub(r'<nav[^>]*>.*?</nav>', '', block, flags=re.DOTALL | re.IGNORECASE)
        block = re.sub(r'<aside[^>]*>.*?</aside>', '', block, flags=re.DOTALL | re.IGNORECASE)
        block = re.sub(r'<footer[^>]*>.*?</footer>', '', block, flags=re.DOTALL | re.IGNORECASE)

        # Keep only content tags
        allowed_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li',
                       'blockquote', 'pre', 'code', 'em', 'strong', 'b', 'i', 'img', 'a', 'br']

        # This is a simplified approach - keeps most content
        return block

    def _extract_title_from_content(self, html):
        """Extract title from content headers."""
        for i in range(1, 7):
            match = re.search(f'<h{i}[^>]*>(.*?)</h{i}>', html, re.IGNORECASE | re.DOTALL)
            if match:
                title = self._strip_tags(match.group(1))
                return self._decode_html_entities(title)
        return ''

    def _html_to_markdown(self, html, base_url=''):
        """Convert HTML to Markdown."""
        md = html

        # Headers
        for i in range(6, 0, -1):
            md = re.sub(f'<h{i}[^>]*>(.*?)</h{i}>', f'\n{"#" * i} \\1\n', md, flags=re.IGNORECASE | re.DOTALL)

        # Bold and italic
        md = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', md, flags=re.IGNORECASE | re.DOTALL)
        md = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', md, flags=re.IGNORECASE | re.DOTALL)

        # Links
        def replace_link(match):
            text = self._strip_tags(match.group(2))
            href = re.search(r'href\s*=\s*["\']([^"\']+)["\']', match.group(1))
            if href:
                url = href.group(1)
                if base_url and not url.startswith('http'):
                    url = urljoin(base_url, url)
                return f'[{text}]({url})'
            return text

        md = re.sub(r'<a([^>]*)>(.*?)</a>', replace_link, md, flags=re.IGNORECASE | re.DOTALL)

        # Images
        def replace_img(match):
            alt = re.search(r'alt\s*=\s*["\']([^"\']+)["\']', match.group(0))
            src = re.search(r'src\s*=\s*["\']([^"\']+)["\']', match.group(0))
            if src:
                url = src.group(1)
                if base_url and not url.startswith('http'):
                    url = urljoin(base_url, url)
                alt_text = alt.group(1) if alt else ''
                return f'\n![{alt_text}]({url})\n'
            return ''

        md = re.sub(r'<img[^>]*>', replace_img, md, flags=re.IGNORECASE)

        # Lists
        md = re.sub(r'<ul[^>]*>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'</ul>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'<ol[^>]*>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'</ol>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', md, flags=re.IGNORECASE | re.DOTALL)

        # Blockquotes
        md = re.sub(r'<blockquote[^>]*>(.*?)</blockquote>', r'\n> \1\n', md, flags=re.IGNORECASE | re.DOTALL)

        # Code
        md = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', md, flags=re.IGNORECASE | re.DOTALL)
        md = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', md, flags=re.IGNORECASE | re.DOTALL)

        # Paragraphs and breaks
        md = re.sub(r'<p[^>]*>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'</p>', '\n', md, flags=re.IGNORECASE)
        md = re.sub(r'<br[^>]*>', '\n', md, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        md = self._strip_tags(md)

        # 1. Replace multiple sequences of spaces/tabs/non-breaking spaces (including at start/end of lines) with a single space.
        md = re.sub(r'[ \t\xa0]{2,}', ' ', md)

        # 2. Trim whitespace from the start/end of each line.
        md = re.sub(r'^[ \t\xa0]+|[ \t\xa0]+$', '', md, flags=re.MULTILINE)

        # Clean up newlines (already present in your code)
        md = re.sub(r'\n{3,}', '\n\n', md)
        md = self._decode_html_entities(md)

        return md.strip()

    def _strip_tags(self, html):
        """Remove all HTML tags."""
        return re.sub(r'<[^>]+>', '', html)

    def _decode_html_entities(self, text):
        """Decode common HTML entities."""
        entities = {
            '&nbsp;': ' ', '&lt;': '<', '&gt;': '>', '&amp;': '&',
            '&quot;': '"', '&#39;': "'", '&apos;': "'",
            '&mdash;': '—', '&ndash;': '–', '&rsquo;': '’', # Changed here
            '&lsquo;': '‘', # Changed here
            '&rdquo;': '"', '&ldquo;': '"',
            # Add a fix for the malformed entity if it persists
            "\\'&lsquo;\\': ": "'", # New line to fix the specific malformation
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)

        # Decode numeric entities
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)

        return text

# Create a module-level function for easy importing
def parse(url='', html='', output='markdown', timeout=15):
    """
    Extract article content from URL or HTML.

    Usage:
        title, content = WebParser().parse(url='https://example.com/article', output='markdown', timeout=15)

    Args:
        url: Article URL to fetch
        html: Pre-fetched HTML content (optional)
        output: Output format ('markdown' or 'html')
        timeout: Request timeout in seconds

    Returns:
        Tuple of (title, content)
    """
    parser = WebParser()
    return parser.parse(url=url, html=html, output=output, timeout=timeout)