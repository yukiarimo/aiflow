import os
import json
import re
import time
import datetime
import requests
import jwt
import smtplib
import imaplib
import email
import email.utils
import caldav
import pytz
import yaml
import html
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, quote
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import hashlib
from PIL import Image
from io import BytesIO


def get_env(key, default=None):
	return os.environ.get(key, default)


class ChatHistoryManager:
	def __init__(self, file_path="db/chat.json"):
		self.file_path = file_path
		self._ensure_file()

	def _ensure_file(self):
		if not os.path.exists(self.file_path):
			with open(self.file_path, "w", encoding="utf-8") as f:
				json.dump([], f)

	def load_history(self):
		try:
			with open(self.file_path, "r", encoding="utf-8") as f:
				return json.load(f)
		except:
			return []

	def save_history(self, history):
		with open(self.file_path, "w", encoding="utf-8") as f:
			json.dump(history, f, indent=2)

	def add_message(self, role, text, images=None):
		history = self.load_history()
		msg = {"id": f"msg-{int(time.time() * 1000)}", "name": role, "text": text, "images": images or [], "timestamp": time.time(), }
		history.append(msg)
		self.save_history(history)
		return msg

	def clear_history(self):
		self.save_history([])


def get_config(config_path="static/config.json", config=None):
	default_config = {"yuna": {"batch_size": 512, "bos": ["<|endoftext|>", True], "context_length": 16384, "flash_attn": True, "gpu_layers": -1, "kokoro": False, "last_n_tokens_size": 128, "max_new_tokens": 1024, "offload_kqv": True, "repetition_penalty": 1.1, "seed": -1, "stop": ["<memory>", "</memory>", "<shujinko>", "</shujinko>", "<aibo>", "</aibo>", "<dialog>", "</dialog>", "<yuki>", "</yuki>", "<yuna>", "</yuna>", "<hito>", "</hito>", "<qt>", "</qt>", "<action>", "</action>", "<data>", "</data>"], "temperature": 0.7, "threads": 8, "top_k": 40, "top_p": 0.9, "use_mlock": True, "use_mmap": True, }, "server": {"yuna_speech_mode": "hanasu", "yuna_speech_model": ["/Users/yuki/Documents/Github/aiflow/aiflow/models/hanasu/config.json", "/Users/yuki/Documents/Github/yuna-ai/lib/models/hanasu/G_6000.pth"], "yuna_audio_model": "/Users/yuki/Documents/Github/yuna-ai/lib/models/audio/qwen3-asr-mlx", "yuna_audio_mode": "yuna_audio", "yuna_text_model": "/Users/yuki/Documents/Github/yuna-ai/lib/models/yuna/yuna-ai-v3-miru-loli-mlx", "yuna_text_mode": "yuna_vlm", "sounds": True, }, }

	if not os.path.exists(config_path):
		return default_config
	mode = "r" if config is None else "w"
	with open(config_path, mode) as f:
		return json.load(f) if config is None else json.dump(config, f, indent=4)


def clearText(text):
	text = text[0] if isinstance(text, tuple) else text
	original_text = text
	replacements = [("</yuki>", ""), ("</yuna>", ""), ("<yuki>", ""), ("<yuna>", "")]
	for old, new in replacements:
		text = text.replace(old, new)
	text = re.sub(r"<[^>]+>", "", text)
	text = re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]+", "", text)
	text = re.sub(r":-?\)|:-?\(|;-?\)|:-?D|:-?P", "", text)
	text = " ".join(text.split()).strip()
	if text != original_text:
		return clearText(text)
	return text


def reverse_image_search(image_path, session_token=None):
	"""Perform reverse image search using Kagi API"""
	url = "https://kagi.com/reverse/upload"

	try:
		# Open and read the image file
		with open(image_path, "rb") as image_file:
			files = {"file": image_file}

			cookies = {}  # Add session cookies if provided
			headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Referer": "https://kagi.com/images", }

			if session_token:
				cookies["kagi_session"] = session_token

			# Send POST request with the image
			response = requests.post(url, files=files, cookies=cookies, headers=headers, allow_redirects=True)

			# Check if we got redirected to login or signin
			if "login" in response.url or "signin" in response.url:
				print("\n❌ Error: Authentication failed.")
				print("The session token may be expired or invalid.")
				return

			# Check if we got a successful redirect to image results
			if "/images?" in response.url and "reverse=upload" in response.url:
				print("\n✓ Image uploaded successfully!")
				print(f"🔗 View results in browser: {response.url}")

				# Parse the HTML to extract image results
				soup = BeautifulSoup(response.content, "html.parser")

				# Try to find image results
				print("\n📸 Similar images found on the page:")

				# Look for the page title
				title = soup.find("title")
				if title: print(f"Page title: {title.text}")

				image_results = soup.find_all("div", class_="_0_image_item")

				if image_results:
					print(f"\nFound {len(image_results)} similar images:")
					for i, item in enumerate(image_results, 1):
						img_tag = item.find("img", class_="_0_img_src")
						if not img_tag:
							continue

						title = item.get("data-title", "N/A")
						# The src is a relative proxy URL, prepend the domain
						proxy_img_url = "https://kagi.com" + img_tag.get("src", "")
						source_url = item.get("data-host_url", "N/A")
						width = item.get("data-width", "N/A")
						height = item.get("data-height", "N/A")

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
		self.title = ""
		self.content = ""

	def parse(self, url="", html="", output="markdown", timeout=15):
		"""Extract article content from URL or HTML"""
		if not html and url:
			html = self._download(url, timeout)

		if not html:
			return "", ""

		# Clean HTML
		html = self._clean_html(html)

		# Extract metadata
		meta = self._extract_metadata(html)
		title = meta.get("title", "")

		# Find main content
		content_html = self._extract_content(html)

		if not content_html:
			return title, ""

		# Extract title from content if not found in metadata
		if not title:
			title = self._extract_title_from_content(content_html)

		# Convert to requested format
		if output == "markdown":
			content = self._html_to_markdown(content_html, url)
		else:
			content = content_html

		# Clean up content
		text_only = self._strip_tags(content)
		if len(text_only.strip()) < 200:
			return "", ""

		return title.strip(), content.strip()

	def _download(self, url, timeout: int):
		"""Download HTML from URL."""
		try:
			headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
			response = requests.get(url, timeout=timeout, headers=headers)
			response.raise_for_status()

			# Handle encoding
			if response.encoding == "ISO-8859-1":
				response.encoding = response.apparent_encoding

			return response.text
		except Exception:
			return ""

	def _clean_html(self, html):
		"""Remove unwanted elements from HTML."""
		# Remove comments
		html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

		# Remove script and style tags
		html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
		html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
		html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)

		# Remove hidden elements
		html = re.sub(r'<[^>]+style\s*=\s*["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>.*?</[^>]+>', "", html, flags=re.DOTALL | re.IGNORECASE)

		return html

	def _extract_metadata(self, html):
		"""Extract metadata from HTML."""
		meta = {}

		# Title from <title> tag
		title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
		if title_match:
			title = title_match.group(1)
			title = re.sub(r"<[^>]+>", "", title)
			title = title.split("|")[0].split("-")[0].strip()
			meta["title"] = self._decode_html_entities(title)

		return meta

	def _extract_content(self, html):
		"""Extract main article content using text density analysis."""
		# Find body content
		body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL)
		if not body_match:
			return ""
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

		return ""

	def _split_into_blocks(self, html) -> list:
		"""Split HTML into content blocks."""
		# Find major container tags
		containers = re.findall(r"<(div|article|section|main)[^>]*>(.*?)</\1>", html, re.IGNORECASE | re.DOTALL)
		blocks = [match[1] for match in containers]

		if not blocks:
			blocks = [html]

		return blocks

	def _score_block(self, block) -> float:
		"""Score a content block based on text density and content signals."""
		text = self._strip_tags(block)  # Remove tags for text analysis
		text_len = len(text.strip())

		if text_len < 100:
			return 0

		# Count elements
		p_count = len(re.findall(r"<p[^>]*>", block, re.IGNORECASE))
		link_count = len(re.findall(r"<a[^>]*>", block, re.IGNORECASE))
		img_count = len(re.findall(r"<img[^>]*>", block, re.IGNORECASE))

		# Calculate score
		score = text_len
		score += p_count * 50  # Paragraphs are good
		score -= link_count * 20  # Too many links = navigation
		score += min(img_count * 30, 150)  # Images are good but cap it

		# Check for article keywords
		if re.search(r"(article|post|entry|content|main|story)", block, re.IGNORECASE):
			score += 100

		# Penalize navigation patterns
		if re.search(r"(nav|menu|sidebar|footer|header|comment)", block, re.IGNORECASE):
			score -= 200

		return score

	def _clean_content_block(self, block):
		"""Clean extracted content block."""
		# Remove navigation elements
		block = re.sub(r"<nav[^>]*>.*?</nav>", "", block, flags=re.DOTALL | re.IGNORECASE)
		block = re.sub(r"<aside[^>]*>.*?</aside>", "", block, flags=re.DOTALL | re.IGNORECASE)
		block = re.sub(r"<footer[^>]*>.*?</footer>", "", block, flags=re.DOTALL | re.IGNORECASE)

		# This is a simplified approach - keeps most content
		return block

	def _extract_title_from_content(self, html):
		"""Extract title from content headers."""
		for i in range(1, 7):
			match = re.search(f"<h{i}[^>]*>(.*?)</h{i}>", html, re.IGNORECASE | re.DOTALL)
			if match:
				title = self._strip_tags(match.group(1))
				return self._decode_html_entities(title)
		return ""

	def _html_to_markdown(self, html, base_url=""):
		"""Convert HTML to Markdown."""
		md = html

		# Headers
		for i in range(6, 0, -1):
			md = re.sub(f"<h{i}[^>]*>(.*?)</h{i}>", f"\n{'#' * i} \\1\n", md, flags=re.IGNORECASE | re.DOTALL)

		# Bold and italic
		md = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", md, flags=re.IGNORECASE | re.DOTALL)

		# Links
		def replace_link(match):
			text = self._strip_tags(match.group(2))
			href = re.search(r'href\s*=\s*["\']([^"\']+)["\']', match.group(1))
			if href:
				url = href.group(1)
				if base_url and not url.startswith("http"):
					url = urljoin(base_url, url)
				return f"[{text}]({url})"
			return text

		md = re.sub(r"<a([^>]*)>(.*?)</a>", replace_link, md, flags=re.IGNORECASE | re.DOTALL)

		# Images
		def replace_img(match):
			alt = re.search(r'alt\s*=\s*["\']([^"\']+)["\']', match.group(0))
			src = re.search(r'src\s*=\s*["\']([^"\']+)["\']', match.group(0))
			if src:
				url = src.group(1)
				if base_url and not url.startswith("http"):
					url = urljoin(base_url, url)
				alt_text = alt.group(1) if alt else ""
				return f"\n![{alt_text}]({url})\n"
			return ""

		md = re.sub(r"<img[^>]*>", replace_img, md, flags=re.IGNORECASE)

		# Lists
		md = re.sub(r"<ul[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</ul>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<ol[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</ol>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", md, flags=re.IGNORECASE | re.DOTALL)

		# Blockquotes
		md = re.sub(r"<blockquote[^>]*>(.*?)</blockquote>", r"\n> \1\n", md, flags=re.IGNORECASE | re.DOTALL)

		# Code
		md = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", md, flags=re.IGNORECASE | re.DOTALL)

		# Paragraphs and breaks
		md = re.sub(r"<p[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</p>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<br[^>]*>", "\n", md, flags=re.IGNORECASE)

		# Remove remaining HTML tags
		md = self._strip_tags(md)

		# 1. Replace multiple sequences of spaces/tabs/non-breaking spaces (including at start/end of lines) with a single space.
		md = re.sub(r"[ \t\xa0]{2,}", " ", md)

		# 2. Trim whitespace from the start/end of each line.
		md = re.sub(r"^[ \t\xa0]+|[ \t\xa0]+$", "", md, flags=re.MULTILINE)

		# Clean up newlines (already present in your code)
		md = re.sub(r"\n{3,}", "\n\n", md)
		md = self._decode_html_entities(md)

		return md.strip()

	def _strip_tags(self, html):
		"""Remove all HTML tags."""
		return re.sub(r"<[^>]+>", "", html)

	def _decode_html_entities(self, text):
		"""Decode common HTML entities."""
		entities = {"&nbsp;": " ", "&lt;": "<", "&gt;": ">", "&amp;": "&", "&quot;": '"', "&#39;": "'", "&apos;": "'", "&mdash;": "—", "&ndash;": "–", "&rsquo;": "’", "&lsquo;": "‘", "&rdquo;": '"', "&ldquo;": '"', "\\'&lsquo;\\': ": "'"}
		for entity, char in entities.items():
			text = text.replace(entity, char)

		# Decode numeric entities
		text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
		text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)

		return text


# Create a module-level function for easy importing
def parse(url="", html="", output="markdown", timeout=15):
	"""Extract article content from URL or HTML"""
	parser = WebParser()
	return parser.parse(url=url, html=html, output=output, timeout=timeout)


def parse_rss(url, limit=None, as_json=False):
	"""Fetches, sanitizes, parses, and formats an RSS feed from a URL"""
	# FETCH & SANITIZE
	try:
		headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"}
		response = requests.get(url, headers=headers, timeout=15)
		response.raise_for_status()

		# Decode and ignore errors
		content = response.content.decode("utf-8", errors="ignore")

		# Sanitize: Remove illegal XML control characters (fixes invalid token errors)
		xml_content = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", content)

	except Exception as e:
		return f"Error: Failed to fetch feed: {e}"

	# PARSE XML
	if not xml_content:
		return ""

	try:
		root = ET.fromstring(xml_content)
	except ET.ParseError as e:
		return f"Error: Invalid XML: {e}"

	channel = root.find("channel")
	if channel is None:
		return "Error: Feed does not contain <channel> (might be Atom format)"

	def clean(text):
		return html.unescape(text.strip()) if text else ""

	# Channel fields
	feed_data = {"title": clean(channel.findtext("title")), "link": clean(channel.findtext("link")), "lastBuildDate": clean(channel.findtext("lastBuildDate")), "pubDate": clean(channel.findtext("pubDate")), "language": clean(channel.findtext("language")), "category": [clean(c.text) for c in channel.findall("category") if c.text], "managingEditor": clean(channel.findtext("managingEditor")), "description": clean(channel.findtext("description")), "items": [], }

	# Item fields
	for item in channel.findall("item"):
		item_data = {"title": clean(item.findtext("title")), "author": clean(item.findtext("author") or item.findtext("dc:creator")), "pubDate": clean(item.findtext("pubDate")), "link": clean(item.findtext("link")), "category": [clean(c.text) for c in item.findall("category") if c.text], "description": clean(item.findtext("description")), }
		# Filter out empty fields
		item_data = {k: v for k, v in item_data.items() if v and v != []}
		feed_data["items"].append(item_data)

	# FORMAT OUTPUT
	items = feed_data.get("items", [])
	if limit is not None:
		items = items[:limit]

	feed_data["items"] = items

	# JSON Mode
	if as_json:
		return json.dumps(feed_data, indent=2, ensure_ascii=False)

	# Text Mode
	output = []
	if feed_data.get("title"):
		output.append(f"Feed: {feed_data['title']}")
	if feed_data.get("link"):
		output.append(f"Link: {feed_data['link']}")
	if feed_data.get("lastBuildDate"):
		output.append(f"Last Build Date: {feed_data['lastBuildDate']}")

	output.append("")

	for idx, item in enumerate(items):
		item_lines = []
		if "title" in item:
			item_lines.append(f"Title: {item['title']}")
		if "author" in item:
			item_lines.append(f"Author: {item['author']}")
		if "pubDate" in item:
			item_lines.append(f"Published: {item['pubDate']}")
		if "link" in item:
			item_lines.append(f"Link: {item['link']}")
		if "description" in item:
			# Simple cleanup for console view
			desc_preview = item["description"][:500].replace("\n", " ") + "..."
			item_lines.append("")
			item_lines.append(f"Summary: {desc_preview}")

		output.append("\n".join(item_lines))

		# Separator
		if idx < len(items) - 1:
			output.append("-" * 60)

	return "\n".join(output)


# Yuna news
class Yuna_News:
	def __init__(self):
		# Paths
		self.intel_dir = "db"
		self.archive_dir = os.path.join(self.intel_dir, "archive")
		self.image_dir = os.path.join(self.intel_dir, "images")
		self.state_file = os.path.join(self.intel_dir, "user_state.json")

		for d in [self.archive_dir, self.image_dir]:
			os.makedirs(d, exist_ok=True)

		self._ensure_state()
		self.kagi_root = "https://news.kagi.com"
		self.kite_url = f"{self.kagi_root}/kite.json"

		# Exact RSS List
		self.rss_feeds = ["https://www.nature.com/nature.rss", "https://www.sciencedaily.com/rss/all.xml", "https://scitechdaily.com/feed/", "https://www.universetoday.com/feed/", "https://www.nasa.gov/rss/dyn/lg_image_of_the_day.rss", "https://www.nasa.gov/rss/dyn/breaking_news.rss", "https://rss.beehiiv.com/feeds/CHXHRVUx6h.xml", "https://rss.beehiiv.com/feeds/Vy37NcFo03.xml", "https://rss.beehiiv.com/feeds/h1qs4tUVIj.xml", "https://www.technologyreview.com/feed/", "https://www.quantamagazine.org/feed/", "https://feeds.arstechnica.com/arstechnica/technology-lab", "https://feeds.arstechnica.com/arstechnica/science", "https://feeds.arstechnica.com/arstechnica/apple", ]

	def _ensure_state(self):
		if not os.path.exists(self.state_file):
			with open(self.state_file, "w") as f:
				json.dump({"read": [], "bookmarks": []}, f)

	def _get_state(self):
		try:
			with open(self.state_file, "r") as f:
				return json.load(f)
		except:
			return {"read": [], "bookmarks": []}

	def _save_state(self, state):
		with open(self.state_file, "w") as f:
			json.dump(state, f)

	def set_action(self, sid, action, value):
		state = self._get_state()
		if value and sid not in state[action]:
			state[action].append(sid)
		elif not value and sid in state[action]:
			state[action].remove(sid)
		self._save_state(state)
		return True

	def _cache_image(self, url):
		"""Downloads, compresses, and caches images."""
		if not url or not isinstance(url, str) or not url.startswith("http"):
			return None
		try:
			fname = f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
			fpath = os.path.join(self.image_dir, fname)
			if os.path.exists(fpath):
				return fname

			# Mimic browser to avoid 403
			headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", "Referer": "https://kagi.com/", }
			r = requests.get(url, headers=headers, timeout=5)
			if r.status_code == 200:
				img = Image.open(BytesIO(r.content))
				if img.mode in ("RGBA", "P"):
					img = img.convert("RGB")
				if img.width > 1000:
					ratio = 1000 / float(img.width)
					img = img.resize((1000, int(img.height * ratio)), Image.LANCZOS)
				img.save(fpath, "JPEG", quality=70, optimize=True)
				return fname
		except:
			pass
		return None

	def sync_daily(self):
		"""Archives all requested categories and RSS feeds."""
		stats = {"yuna": 0, "rss": 0}

		# THE FULL REQUESTED LIST
		my_cats = ["World", "USA", "Business", "Technology", "Science", "Sports", "Gaming", "3D Printing", "AI", "Apple", "Asia", "Canada", "China", "Cryptocurrency", "Cybersecurity", "Google", "Italy", "Japan", "Linux & OSS", "Microsoft", "Movies", "Music", "Professional Wrestling", "Romania", "Soccer", "South Korea", "Taiwan", "UK", ]

		try:
			# 1. Yuna Network (Kagi)
			kite = requests.get(self.kite_url).json()
			for cat in kite.get("categories", []):
				if cat["name"] not in my_cats:
					continue

				data = requests.get(f"{self.kagi_root}/{cat['file']}").json()
				f_name = (f"{cat['name'].lower().replace(' ', '_').replace('&', 'and')}.json")
				f_path = os.path.join(self.archive_dir, f_name)

				existing = []
				if os.path.exists(f_path):
					with open(f_path, "r") as f:
						existing = json.load(f)

				e_ids = {c.get("_id") for c in existing}
				count_added = 0

				for c in data.get("clusters", []):
					sid = f"kagi_{c.get('timestamp')}_{hashlib.md5(c['title'].encode()).hexdigest()[:8]}"
					if sid in e_ids:
						continue

					# Safe Image Extraction
					img_u = None
					if c.get("primary_image"):
						if isinstance(c["primary_image"], dict):
							img_u = c["primary_image"].get("url")
						else:
							img_u = c["primary_image"]

					entry = {"_id": sid, "_type": "yuna", "title": c.get("title"), "short_summary": c.get("short_summary"), "timestamp": c.get("timestamp"), "category": c.get("category"), "talking_points": c.get("talking_points", []), "scientific_significance": c.get("scientific_significance", []), "perspectives": c.get("perspectives", []), "unique_domains": c.get("unique_domains", 1), "primary_image": c.get("primary_image"), "_local_img": self._cache_image(img_u), }
					existing.append(entry)
					count_added += 1

				if count_added > 0:
					existing.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
					with open(f_path, "w") as f:
						json.dump(existing, f, indent=2)
					stats["yuna"] += count_added

			# 2. RSS Sync (Aggressive)
			rss_path = os.path.join(self.archive_dir, "personal_stream.json")
			existing_rss = []
			if os.path.exists(rss_path):
				with open(rss_path, "r") as f:
					existing_rss = json.load(f)

			r_ids = {item["_id"] for item in existing_rss}

			# Common Namespaces
			ns = {"media": "http://search.yahoo.com/mrss/", "content": "http://purl.org/rss/1.0/modules/content/", "atom": "http://www.w3.org/2005/Atom", }

			for url in self.rss_feeds:
				try:
					r = requests.get(url, timeout=10)
					# Strip null bytes that break parsing
					clean_content = r.text.replace("\x00", "")
					root = ET.fromstring(clean_content.encode("utf-8"))

					items = root.findall(".//item") or root.findall(".//atom:entry", ns)
					source = (urlparse(url).netloc.replace("www.", "").split(".")[0].upper())

					for i in items[:10]:
						# Title
						title = (i.findtext("title") or i.findtext("{http://www.w3.org/2005/Atom}title", namespaces=ns) or "Untitled").strip()

						# Link
						link = i.findtext("link")
						if not link:
							link_node = i.find("atom:link", ns)
							if link_node is not None:
								link = link_node.get("href")
						if not link:
							continue

						sid = f"rss_{hashlib.md5(link.encode()).hexdigest()}"
						if sid in r_ids:
							continue

						# Description / Content (Aggressive)
						desc = i.findtext("{http://purl.org/rss/1.0/modules/content/}encoded", namespaces=ns)
						if not desc:
							desc = i.findtext("description")
						if not desc:
							desc = i.findtext("atom:summary", namespaces=ns)
						if not desc:
							desc = ""

						# Strip HTML for snippet
						snippet = (re.sub(r"<[^>]+>", "", desc).replace("\n", " ").strip()[:350] + "...")

						# Image Hunting
						thumb = None
						mc = i.find("media:content", ns)
						if mc is not None:
							thumb = mc.get("url")
						if not thumb:
							enc = i.find("enclosure")
							if enc is not None:
								thumb = enc.get("url")
						if not thumb:
							# Regex hunt in HTML content
							img_match = re.search(r'<img.*?src=["\'](.*?)["\']', desc)
							if img_match:
								thumb = img_match.group(1)

						pubd = i.findtext("pubDate") or i.findtext("dc:date", namespaces=ns)
						if not pubd:
							pubd = i.findtext("{http://www.w3.org/2005/Atom}updated") or i.findtext("{http://www.w3.org/2005/Atom}published")

						stamp = time.time()
						if pubd:
							try:
								dt = email.utils.parsedate_to_datetime(pubd)
								if dt is not None:
									stamp = dt.timestamp()
							except:
								try:
									import datetime
									stamp = datetime.datetime.fromisoformat(pubd.replace("Z", "+00:00")).timestamp()
								except:
									pass

						existing_rss.append({"_id": sid, "_type": "rss", "source": source, "title": title, "url": link, "snippet": snippet, "timestamp": stamp, "_local_img": self._cache_image(thumb), })
						stats["rss"] += 1
				except Exception as e:
					print(f"Skipping RSS {url}: {e}")
					continue

			existing_rss.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
			with open(rss_path, "w") as f:
				json.dump(existing_rss[:1000], f, indent=2)

			return {"status": "success", "stats": stats}
		except Exception as e:
			return {"error": str(e)}

	def get_feed(self, mode="yuna", cat=None, search=None, day=None):
		"""Loads from local archive files."""
		state = self._get_state()
		results = []

		# Select files
		if mode == "rss":
			files = ["personal_stream.json"]
		elif mode == "all":
			files = [f for f in os.listdir(self.archive_dir) if f.endswith(".json")]
		else:
			files = [f for f in os.listdir(self.archive_dir) if f.endswith(".json") and f != "personal_stream.json"]

		for f in files:
			# Category match (e.g. "linux_oss.json")
			if cat:
				clean_cat = cat.lower().replace(" ", "_").replace("&", "and")
				if clean_cat not in f:
					continue

			p = os.path.join(self.archive_dir, f)
			if not os.path.exists(p):
				continue

			try:
				with open(p, "r") as j:
					for s in json.load(j):
						# Search Filter
						content = (str(s.get("title", "")) + " " + str(s.get("short_summary", "")) + " " + str(s.get("snippet", ""))).lower()
						if search and search.lower() not in content:
							continue

						# Date Filter
						if day == "today":
							ts = s.get("timestamp") or 0
							if (datetime.datetime.fromtimestamp(ts).date() != datetime.date.today()):
								continue

						# State Injection
						s["_read"] = s["_id"] in state["read"]
						s["_bookmarked"] = s["_id"] in state["bookmarks"]
						results.append(s)
			except:
				continue

		results.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
		return results

	def get_saved(self):
		state = self._get_state()
		all_data = self.get_feed(mode="all")
		return {"bookmarks": [s for s in all_data if s["_bookmarked"]], "history": [s for s in all_data if s["_read"]], }

	def get_world_news(self, query):
		key = os.environ.get("API_KAGI_KEY")
		if not key:
			return []
		try:
			headers = {"Authorization": f"Bot {key}"}
			r = requests.get("https://kagi.com/api/v0/enrich/news", params={"q": query}, headers=headers, timeout=10)
			return r.json().get("data", [])
		except:
			return []


# Yuna calendar
class Yuna_Calendar:
	def __init__(self):
		self.user = get_env("EMAIL_YUKI")
		self.password = get_env("APPLE_APP_PASSWORD_YUKI")
		self.url = "https://caldav.icloud.com/"
		self.tz = pytz.timezone("America/Edmonton")

	def get_events(self, start_date=None, end_date=None):
		if not start_date:
			start_date = datetime.datetime.now(self.tz)
		if not end_date:
			end_date = start_date + datetime.timedelta(days=1)

		try:
			client = caldav.DAVClient(url=self.url, username=self.user, password=self.password)
			principal = client.principal()
			results = []
			for cal in principal.calendars():
				events = cal.search(start=start_date, end=end_date, event=True, expand=True)
				for e in events:
					ical = e.icalendar_component
					start_dt = ical.get("DTSTART").dt
					if isinstance(start_dt, datetime.datetime):
						start_dt = start_dt.astimezone(self.tz)
					elif isinstance(start_dt, datetime.date):
						start_dt = self.tz.localize(datetime.datetime(start_dt.year, start_dt.month, start_dt.day))
					results.append({"title": str(ical.get("SUMMARY")), "start": start_dt.isoformat(), "location": str(ical.get("LOCATION", "")), "calendar": str(cal.url).split("/")[-2], })
			results.sort(key=lambda x: x["start"])
			return results
		except Exception as e:
			return {"error": str(e)}


# YUNA EMAIL
class Yuna_Email:
	def __init__(self):
		self.email = get_env("EMAIL_YUNA")
		self.user_email = get_env("EMAIL_YUKI")
		self.password = get_env("GOOGLE_APP_PASSWORD_YUNA")
		self.imap_server = "imap.gmail.com"
		self.smtp_server = "smtp.gmail.com"

	def _connect_imap(self):
		mail = imaplib.IMAP4_SSL(self.imap_server)
		mail.login(self.email, self.password)
		return mail

	def get_emails(self, folder="INBOX", limit=20, unread_only=False):
		try:
			mail = self._connect_imap()
			mail.select(folder)

			criterion = "UNSEEN" if unread_only else "ALL"
			_, data = mail.search(None, criterion)

			if not data[0]:
				return []

			ids = data[0].split()
			latest_ids = ids[-limit:]  # Get last N

			results = []
			for eid in reversed(latest_ids):
				_, msg_data = mail.fetch(eid, "(RFC822)")
				raw = email.message_from_bytes(msg_data[0][1])

				# Decode Subject
				subject, encoding = decode_header(raw["Subject"])[0]
				if isinstance(subject, bytes):
					subject = subject.decode(encoding or "utf-8")

				# Get Body
				body = ""
				if raw.is_multipart():
					for part in raw.walk():
						if part.get_content_type() == "text/plain":
							body = part.get_payload(decode=True).decode(errors="ignore")
							break
				else:
					body = raw.get_payload(decode=True).decode(errors="ignore")

				results.append({"id": eid.decode(), "msg_id": raw.get("Message-ID"), "from": raw.get("From"), "subject": subject, "date": raw.get("Date"), "body_preview": body[:200], })

			mail.close()
			mail.logout()
			return results
		except Exception as e:
			return {"error": str(e)}

	def send_email(self, to, subject, body, attachment_path=None, reply_to_id=None):
		try:
			msg = MIMEMultipart()
			msg["From"] = self.email
			msg["To"] = to or self.user_email
			msg["Subject"] = subject

			if reply_to_id:
				msg["In-Reply-To"] = reply_to_id
				msg["References"] = reply_to_id

			msg.attach(MIMEText(body, "plain"))

			if attachment_path:
				filename = os.path.basename(attachment_path)
				with open(attachment_path, "rb") as f:
					part = MIMEBase("application", "octet-stream")
					part.set_payload(f.read())
				encoders.encode_base64(part)
				part.add_header("Content-Disposition", f"attachment; filename= {filename}")
				msg.attach(part)

			server = smtplib.SMTP_SSL(self.smtp_server, 465)
			server.login(self.email, self.password)
			server.send_message(msg)
			server.quit()
			return {"status": "sent"}
		except Exception as e:
			return {"error": str(e)}

	def delete_email(self, email_id):
		try:
			mail = self._connect_imap()
			mail.select("INBOX")
			# Mark as deleted
			mail.store(email_id, "+FLAGS", "\\Deleted")
			# Expunge (Permanently remove)
			mail.expunge()
			mail.close()
			mail.logout()
			return {"status": "deleted"}
		except Exception as e:
			return {"error": str(e)}

	def forward_email(self, email_id, to_address):
		# Fetch original, create new message with FW: subject and original body
		# (Simplified logic)
		return self.send_email(to_address, "FW: ...", "Forwarded message...")


# YUNA WEATHER
class Yuna_Weather:
	def __init__(self):
		self.team_id = get_env("APPLE_TEAM_ID")
		self.key_id = get_env("APPLE_KEY_ID")
		self.client_id = get_env("APPLE_CLIENT_ID", "com.yuna-ai.app")
		self.private_key_path = get_env("APPLE_PRIVATE_KEY_PATH")

	def _get_token(self, scope="weather"):
		if not self.private_key_path:
			return None
		with open(self.private_key_path, "r") as f:
			pk = f.read()

		now = int(time.time())
		payload = {"iss": self.team_id, "iat": now, "exp": now + 3600}
		if scope == "weather" or "map":
			payload["sub"] = self.client_id

		return jwt.encode(payload=payload, key=pk, algorithm="ES256", headers={"kid": self.key_id, "id": f"{self.team_id}.{self.client_id}"} if scope == "weather" else {"kid": self.key_id, "typ": "JWT"})

	def get_weather(self, city="Calgary"):
		# 1. Geocode (MapKit)
		map_token = self._get_token("map")
		geo_url = f"https://maps-api.apple.com/v1/geocode?q={quote(city)}&lang=en-US"
		geo_resp = requests.get(geo_url, headers={"Authorization": f"Bearer {map_token}"})

		print(geo_resp)

		if geo_resp.status_code != 200:
			return {"error": "Geocode failed"}
		coords = geo_resp.json()["results"][0]["coordinate"]
		lat, lon = coords["latitude"], coords["longitude"]

		# 2. Weather (WeatherKit)
		w_token = self._get_token("weather")
		w_url = f"https://weatherkit.apple.com/api/v1/weather/en/{lat}/{lon}?dataSets=currentWeather,forecastDaily"
		w_resp = requests.get(w_url, headers={"Authorization": f"Bearer {w_token}"})

		return w_resp.json()


# YUNA FILESYSTEM
class Yuna_Filesystem:
	def __init__(self):
		# FIX: Point directly to the Yuna project folder
		self.root_path = os.environ.get("FOLDER_DOWNLOADS_YUNA") or os.getcwd()

	def list_files(self, subpath=""):
		# Cleanup path joining to prevent "Path not found"
		subpath = subpath.strip("/")
		target_dir = os.path.abspath(os.path.join(self.root_path, subpath))

		# Security: stay within root
		if not target_dir.startswith(os.path.abspath(self.root_path)):
			return {"error": "Access Denied"}

		if not os.path.exists(target_dir):
			# Fallback to root if subpath fails
			target_dir = os.path.abspath(self.root_path)

		items = []
		try:
			for entry in os.scandir(target_dir):
				stats = entry.stat()
				# Simplified paths for the frontend
				rel_path = os.path.relpath(entry.path, self.root_path)
				items.append({"name": entry.name, "type": "folder" if entry.is_dir() else "file", "size": stats.st_size, "path": rel_path, })
		except Exception as e:
			return {"error": str(e)}

		return sorted(items, key=lambda x: (x["type"] != "folder", x["name"].lower()))


# LOLI MANAGER
class Yuna_LoliManager:
	def __init__(self):
		self.config_path = "loliconnect.yaml"  # Default

	def get_chains(self):
		if not os.path.exists(self.config_path):
			return {}
		with open(self.config_path, "r") as f:
			try:
				data = yaml.safe_load(f)
				return data.get("chains", {})
			except:
				return {}
