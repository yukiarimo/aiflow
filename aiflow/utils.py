import os
import json
import re
import time
import uuid
import base64
import datetime
import ast
import operator
import requests
import jwt
import email.utils
from zoneinfo import ZoneInfo
import html
import xml.etree.ElementTree as ET
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib.parse import urljoin, quote, urlparse
from bs4 import BeautifulSoup
import hashlib
import traceback
import urllib3
import orjson

YUNA_TLS_VERIFY = False  # Caddy internal CA: macOS trusts it, certifi does not — set to root.crt path to re-enable real verify; urllib3 warning disabled below so Sage logs stay readable
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ACTION_OPEN_RE = re.compile(r"<action>\s*(?P<wrap>[A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)  # body ends at </action> (or next tag / EOS if stop-token ate the closer) — never at the first ")"
ACTION_CLOSE_RE = re.compile(r"</action>", re.IGNORECASE)
ACTION_UNCLOSED_END_RE = re.compile(r"<action>|<data>|</yuna>", re.IGNORECASE)
SKILL_LINE_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*->\s*(?P<args>.*?)\s*$")
_SAGE_TIMER_KEY = "sage_timers"
_SAGE_REMINDER_KEY = "sage_reminders"
_store_locks = {}
_store_locks_guard = threading.Lock()
_MATH_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
SEARCH_URL_DEFAULT = "https://knowledge.yunaai.com"
SEARCH_HOSTS_RETIRED = {"search.yuna-ai.com", "search.yunaai.me"}  # no DNS — repoint at knowledge.yunaai.com
_SEARCH_URL_WARNED = set()
REVERSE_IMAGE_URL = "https://kagi.com/reverse/upload"
REVERSE_IMAGE_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
_NSFW_RE = re.compile(r"(?i)(?<![a-z])(hentai|nhentai|hanime|rule[\s-]?34|\br34\b|pornhub|xvideos|xhamster|xnxx|onlyfans|e621|e-?hentai|exhentai|hitomi\.la|hentaihaven|futanari|yaoi|ecchi|nsfw|\bxxx\b|loli(con)?|shotacon)(?![a-z])")  # Kagi v1 `safe_search` is documented but the account Safe Search toggle wins — same body with true/false/omit. NSFW-off is our gate; NSFW-on uses Bing adlt=off when Kagi returns no adult hosts.
_NSFW_HOSTS = ("nhentai.net", "e-hentai.org", "exhentai.org", "hanime.tv", "hitomi.la", "rule34.xxx", "e621.net", "pornhub.com", "xvideos.com", "xhamster.com", "xnxx.com", "hentaihaven.xxx", "hentai.tv", "spankbang.com", "hentaifoundry.com", "hentai2read.com", "hanime1.me")


def yuna_json_path():
	"""One-file Yuna store. Absolute once YUNA_JSON is set (yuna-ai yuna.py does that)."""
	return os.environ.get("YUNA_JSON") or os.path.join("lib", "yuna.json")


def knowledge_json_path():
	"""Search + News bag. Absolute once YUNA_KNOWLEDGE_JSON is set (yuna-ai knowledge.py)."""
	return os.environ.get("YUNA_KNOWLEDGE_JSON") or os.environ.get("YUNA_SEARCH_JSON") or os.environ.get("YUNA_NEWS_JSON") or os.path.join("lib", "knowledge.json")


def news_json_path():
	return knowledge_json_path()


def dictionary_warm():
	from aiflow.dictionary import warm
	warm()


def sage_today_line():
	tz = ZoneInfo("America/Edmonton")
	now = datetime.datetime.now(tz)
	sat = now.date() + datetime.timedelta(days=(5 - now.weekday()) % 7)
	return f"Today is {now.strftime('%A %Y-%m-%d')} in America/Edmonton. Saturday is {sat.isoformat()}."


def dictionary_lookup(q, lang="", limit=14):
	from aiflow.dictionary import lookup
	return lookup(q, lang=lang, limit=limit)


def get_env(key, default=None):
	return os.environ.get(key, default)


def _dumps_json(data, indent=None):
	"""Serialise ``data``. Compact bags go through ``orjson``; pretty-print keeps stdlib ``indent``."""
	if indent is None:
		return orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")  # orjson returns bytes; OPT_NON_STR_KEYS keeps bag keys that happen to be ints
	return json.dumps(data, indent=indent, ensure_ascii=False)


def _atomic_write_json(path, data, indent=None):
	"""Serialise beside ``path``, fsync, then ``os.replace`` onto it. Opening the destination in ``"w"`` truncates it before a single byte is serialised, so a crash or a full disk mid-write leaves a half-written file — for ``chat.json`` that is the whole conversation. ``os.replace`` is atomic within a filesystem, so every reader sees either the previous file or the complete new one. The temp name is unique so two writers racing on the same destination cannot interleave into one another's buffer."""
	directory = os.path.dirname(path) or "."
	os.makedirs(directory, exist_ok=True)
	tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
	payload = _dumps_json(data, indent=indent)
	with open(tmp, "w", encoding="utf-8") as f:
		f.write(payload)
		f.flush()
		os.fsync(f.fileno())
	os.replace(tmp, path)


def _lock_for(path):
	path = os.path.abspath(path)
	with _store_locks_guard:
		lock = _store_locks.get(path)
		if lock is None:
			lock = threading.Lock()
			_store_locks[path] = lock
		return lock


def bag_load(path):
	"""Whole JSON object at ``path``, or ``{}`` if the file is missing. Corrupt JSON raises."""
	if not os.path.exists(path):
		return {}
	with open(path, "rb") as f:
		raw = f.read()
	if not raw.strip():
		return {}
	data = orjson.loads(raw)
	if not isinstance(data, dict):
		raise ValueError(f"{path}: store must be a JSON object")
	return data


def bag_save(path, data, indent=2):
	_atomic_write_json(path, data, indent=indent)


def bag_get(path, key, default=None):
	with _lock_for(path):
		return bag_load(path).get(key, default)


def bag_set(path, key, value, indent=2):
	with _lock_for(path):
		data = bag_load(path)
		data[key] = value
		bag_save(path, data, indent=indent)
		return data


def bag_update(path, indent=2, **kwargs):
	with _lock_for(path):
		data = bag_load(path)
		data.update(kwargs)
		bag_save(path, data, indent=indent)
		return data


class ChatHistoryManager:
	def __init__(self, file_path=None, key="dialog"):
		self.file_path = file_path or yuna_json_path()
		self.key = key

	@staticmethod
	def new_id():
		"""Return a globally-unique message id. Shape: ``msg-<ms-since-epoch>-<uuid4-hex>``. The uuid4 suffix alone is collision-proof for any realistic lifetime (chatting for 10+ years is nowhere near its 122 bits of entropy); the millisecond prefix just keeps ids roughly sortable and greppable by creation time."""
		return f"msg-{int(time.time() * 1000)}-{uuid.uuid4().hex}"

	def _normalize(self, history):
		"""Guarantee every message carries a unique ``id``, a numeric ``timestamp`` (seconds since epoch) and an ``images`` list. Backfills anything missing — legacy/seed data, or messages the frontend persisted without an id/timestamp — and de-duplicates colliding ids so each message is addressable forever. Missing timestamps are filled with a strictly non-decreasing value so message ordering survives the backfill. Returns ``(history, changed)`` where ``changed`` is True if anything was rewritten (so the caller can heal the on-disk file)."""
		if not isinstance(history, list):
			return [], True
		changed = False
		seen_ids = set()
		last_ts = None
		for msg in history:
			if not isinstance(msg, dict):
				continue
			if not isinstance(msg.get("images"), list):
				msg["images"] = []
				changed = True
			ts = msg.get("timestamp")
			if isinstance(ts, bool) or not isinstance(ts, (int, float)):
				ts = (last_ts + 0.001) if last_ts is not None else time.time()
				msg["timestamp"] = ts
				changed = True
			last_ts = msg["timestamp"]
			mid = msg.get("id")
			if not isinstance(mid, str) or not mid or mid in seen_ids:
				mid = self.new_id()
				msg["id"] = mid
				changed = True
			seen_ids.add(mid)
		return history, changed

	def load_history(self):
		"""No dialog yet means a new transcript; a corrupt store raises. Returning ``[]`` for undecodable JSON would hand the caller an empty conversation and the very next save would overwrite whatever part of the file survived."""
		history = bag_get(self.file_path, self.key, [])
		if not isinstance(history, list):
			raise ValueError(f"{self.file_path}: '{self.key}' must be a list")
		history, changed = self._normalize(history)
		if changed:
			self.save_history(history, normalize=False)
		return history

	def save_history(self, history, normalize=True):
		if normalize:
			history, _ = self._normalize(history)
		bag_set(self.file_path, self.key, history)

	def clear_history(self):
		self.save_history([])


def _check_config(cfg, source):
	"""Reject a config that is not shaped like one. A file that parses but has lost its ``yuna``/``server`` sections degrades silently: memory/shujinko/aibo resolve to nothing, and Yuna answers with an empty system prompt while every request still returns 200."""
	if not isinstance(cfg, dict) or not isinstance(cfg.get("yuna"), dict) or not isinstance(cfg.get("server"), dict):
		raise ValueError(f"{source}: not a Yuna config — needs top-level 'yuna' and 'server' objects")
	return cfg


def get_config(config_path=None, config=None):
	config_path = config_path or yuna_json_path()  # Store is lib/yuna.json (or YUNA_JSON). Extra keys (dialog, memory, great_sage) stay put.
	default_config = {"yuna": {"batch_size": 512, "bos": ["<|endoftext|>", True], "context_length": 16384, "gpu_layers": -1, "kokoro": False, "last_n_tokens_size": 128, "max_new_tokens": 1024, "repetition_penalty": 1.1, "seed": -1, "stop": ["<memory>", "</memory>", "<shujinko>", "</shujinko>", "<aibo>", "</aibo>", "<dialog>", "</dialog>", "<yuki>", "</yuki>", "<yuna>", "</yuna>", "</action>", "<data>", "</data>"], "temperature": 0.7, "threads": 8, "top_k": 40, "top_p": 0.9, }, "server": {"yuna_speech_mode": "vits", "yuna_speech_model": [os.environ.get("YUNA_SPEECH_CONFIG", "lib/models/vits/config.json"), os.environ.get("YUNA_SPEECH_WEIGHTS", "lib/models/vits/G_6000.pth"), os.environ.get("YUNA_SPEECH_BACKEND", "pytorch")], "yuna_audio_model": os.environ.get("YUNA_AUDIO_MODEL", "lib/models/audio/qwen3-asr-mlx"), "yuna_audio_mode": "yuna_audio", "yuna_text_model": os.environ.get("YUNA_TEXT_MODEL", "lib/models/yuna/yuna-ai-v3-miru-loli-mlx"), "yuna_text_mode": "yuna_vlm", "sounds": True, }, }

	if config is not None:
		checked = _check_config(config, config_path)
		bag_update(config_path, indent=4, yuna=checked["yuna"], server=checked["server"])
		return checked
	if not os.path.exists(config_path):
		return default_config
	data = bag_load(config_path)
	return _check_config({"yuna": data.get("yuna"), "server": data.get("server")}, config_path)


def _host_of(url):
	return (url or "").split("://", 1)[-1].split("/", 1)[0].split("@")[-1].split(":", 1)[0].lower()


def nsfw_text(*parts):
	return bool(_NSFW_RE.search(" ".join("" if p is None else str(p) for p in parts)))


def nsfw_host(url):
	h = _host_of(url)
	return any(h == d or h.endswith("." + d) for d in _NSFW_HOSTS)


def query_is_nsfw(query):
	return nsfw_text(query)


def gate_nsfw_results(query, results):
	"""NSFW chip off — empty adult-intent queries; strip adult hosts/terms from the rest."""
	if not isinstance(results, list):
		return []
	if query_is_nsfw(query):
		return []
	out = []
	for it in results:
		if not isinstance(it, dict):
			continue
		blob = " ".join(str(it.get(k) or "") for k in ("title", "snippet", "short_summary", "url", "original_url", "image_url"))
		if nsfw_text(blob) or nsfw_host(it.get("url") or "") or nsfw_host(it.get("original_url") or "") or nsfw_host(it.get("image_url") or ""):
			continue
		out.append(it)
	return out


def _bing_dest(href):
	href = html.unescape(href or "")
	m = re.search(r"[?&]u=a1([^&]+)", href)
	if not m:
		return href if href.startswith("http") and "/ck/" not in href else ""
	raw = m.group(1) + "=" * ((4 - len(m.group(1)) % 4) % 4)
	try:
		out = base64.urlsafe_b64decode(raw.encode()).decode("utf-8", "replace")
	except Exception:
		return ""
	return out if out.startswith("http") else ""


def nsfw_open_search(query, kind, session=None):
	"""Bing `adlt=off` rows in the Knowledge shape. Kagi cannot turn Safe Search off per request."""
	http = session or requests
	q = quote(query)
	headers = {"User-Agent": REVERSE_IMAGE_UA, "Accept-Language": "en-US,en;q=0.9"}
	if kind == "images":
		url = f"https://www.bing.com/images/search?q={q}&adlt=off"
	elif kind == "videos":
		url = f"https://www.bing.com/videos/search?q={q}&adlt=off"
	elif kind == "news":
		url = f"https://www.bing.com/news/search?q={q}&adlt=off"
	else:
		url = f"https://www.bing.com/search?q={q}&adlt=off&count=20"
	try:
		resp = http.get(url, headers=headers, timeout=25, verify=YUNA_TLS_VERIFY)
	except Exception:
		return []
	if not resp.ok:
		return []
	page = resp.text
	rows = []
	if kind == "images":
		for raw in re.findall(r'class="iusc"[^>]+m="([^"]+)"', page):
			try:
				obj = json.loads(html.unescape(raw))
			except Exception:
				continue
			murl, turl = (obj.get("murl") or "").strip(), (obj.get("turl") or "").strip()
			if not (murl or turl):
				continue
			title = html.unescape(obj.get("t") or "")
			page_u = (obj.get("purl") or murl or "").strip()
			rows.append({"title": title, "url": page_u, "image_url": turl or murl, "original_url": murl or turl, "width": 0, "height": 0, "snippet": "", "time": None, "props": {}})
	elif kind == "videos":
		for m in re.finditer(r'ourl="([^"]+)"(.*?)(?=ourl="|$)', page, re.S):
			href, chunk = m.group(1).strip(), m.group(2)[:3000]
			if not href.startswith("http"):
				continue
			alt = re.search(r'alt="([^"]+)"', chunk)
			thumb = re.search(r'data-src-hq="([^"]+)"', chunk) or re.search(r'src="(https://tse[^"]+)"', chunk)
			title = html.unescape(alt.group(1)) if alt else href
			thumb_u = html.unescape(thumb.group(1)) if thumb else ""
			rows.append({"title": title, "url": href, "thumbnail": thumb_u, "duration": "", "publisher": _host_of(href) or "Unknown", "views": None, "likes": None, "snippet": "", "time": None, "props": {}})
	else:
		for block in re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', page, re.S):
			a = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
			if not a:
				continue
			href = _bing_dest(a.group(1))
			if not href:
				continue
			title = re.sub(r"<[^>]+>", "", html.unescape(a.group(2))).strip()
			snip_m = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
			snippet = re.sub(r"<[^>]+>", "", html.unescape(snip_m.group(1))).strip() if snip_m else ""
			host = _host_of(href)
			if kind == "news":
				rows.append({"title": title, "url": href, "publisher": host or "Unknown Publisher", "published_time": "", "snippet": snippet, "image": None, "props": {}, "paywalled": False})
			else:
				rows.append({"t": 0, "url": href, "title": title, "snippet": snippet, "time": "", "image": None, "props": {}, "group_id": host, "paywalled": False})
	seen, out = set(), []
	for it in rows:
		u = (it.get("original_url") or it.get("url") or "").strip()
		if not u or u in seen:
			continue
		seen.add(u)
		out.append(it)
	out = out[:24]
	if kind == "videos" and not out:
		web = nsfw_open_search(query, "web", session=session)
		out = [{"title": it.get("title") or "", "url": it.get("url") or "", "thumbnail": "", "duration": "", "publisher": _host_of(it.get("url") or "") or "Unknown", "views": None, "likes": None, "snippet": it.get("snippet") or "", "time": None, "props": {}} for it in web if it.get("url")]
	return out


def kagi_html_results(query, kind="web", personalized=True, session=None):
	"""Kagi HTML search. v1 `personalizations` is ignored. `personalized=0` is the real off switch on web. News/images/videos treat that query param as extra search words."""
	token = (os.environ.get("KAGI_SESSION_TOKEN") or "").strip()
	if not token:
		return None, "KAGI_SESSION_TOKEN required to disable Personal"
	http = session or requests
	path = {"web": "/html/search", "news": "/html/news", "images": "/html/images", "videos": "/html/videos"}.get(kind, "/html/search")
	params = {"q": query}
	if not personalized and kind == "web":
		params["personalized"] = "0"  # news/images/videos would search for the words "personalized" + "0"
	try:
		resp = http.get("https://kagi.com" + path, params=params, cookies={"kagi_session": token}, headers={"User-Agent": REVERSE_IMAGE_UA}, timeout=30, verify=YUNA_TLS_VERIFY)
	except Exception as e:
		return None, str(e)
	if "login" in resp.url or "signin" in resp.url:
		return None, "Kagi session expired — refresh KAGI_SESSION_TOKEN"
	if not resp.ok:
		return None, resp.reason or str(resp.status_code)
	page = resp.text
	if kind == "images":
		return parse_reverse_image_html(page), None
	rows = []
	if kind == "news":
		for block in re.split(r'<div class="newsResultItem\b', page)[1:]:
			a = re.search(r'<div class="newsResultTitle">\s*<h3[^>]*>\s*<a([^>]*)>(.*?)</a>', block, re.S)
			if not a:
				continue
			href_m = re.search(r'href="([^"]+)"', a.group(1))
			if not href_m:
				continue
			href = href_m.group(1)
			if "kagi.com" in href or "web.archive.org" in href:
				continue
			title_m = re.search(r'title="([^"]+)"', a.group(1))
			title = html.unescape(title_m.group(1) if title_m else re.sub(r"<[^>]+>", "", a.group(2))).strip()
			src_m = re.search(r'aria-label="News source ([^"]+)"', block)
			time_m = re.search(r'class="newsResultTime"[^>]*>\s*([^<]+)', block)
			body_m = re.search(r'class="newsResultBody[^"]*"[^>]*>(.*?)</(?:div|p)>', block, re.S)
			snip = re.sub(r"<[^>]+>", "", html.unescape(body_m.group(1))).strip() if body_m else ""
			rows.append({"title": title, "url": href, "publisher": (src_m.group(1).strip() if src_m else _host_of(href)) or "Unknown Publisher", "published_time": (time_m.group(1).strip() if time_m else ""), "snippet": snip, "image": None, "props": {}, "paywalled": False})
	elif kind == "videos":
		for block in re.split(r'<div class="videoResultItem\b', page)[1:]:
			thumb = re.search(r'<a class="[^"]*videoResultThumbnail[^"]*"([^>]*)>.*?<img[^>]+(?:alt="Video Thumbnail of ([^"]+)")?[^>]*src="([^"]*)"', block, re.S)
			title_a = re.search(r'<a class="[^"]*videoResultTitle[^"]*"([^>]*)>(.*?)</a>', block, re.S)
			href = ""
			if thumb:
				hm = re.search(r'href="([^"]+)"', thumb.group(1))
				href = hm.group(1) if hm else ""
			if title_a and not href:
				hm = re.search(r'href="([^"]+)"', title_a.group(1))
				href = hm.group(1) if hm else ""
			if not href or "kagi.com" in href:
				continue
			title = ""
			if title_a:
				title = re.sub(r"<[^>]+>", "", html.unescape(title_a.group(2))).strip()
			if not title and thumb and thumb.group(2):
				title = html.unescape(thumb.group(2)).strip()
			pub_m = re.search(r'aria-label="Video publisher ([^"]+)"', block)
			dur_m = re.search(r'class="videoResultVideoTime"[^>]*>\s*([^<]+)', block)
			rows.append({"title": title, "url": href, "thumbnail": html.unescape(thumb.group(3)) if thumb else "", "duration": (dur_m.group(1).strip() if dur_m else ""), "publisher": (pub_m.group(1).strip() if pub_m else _host_of(href)) or "Unknown", "views": None, "likes": None, "snippet": "", "time": None, "props": {}})
	else:
		descs = [re.sub(r"<[^>]+>", "", html.unescape(x)).strip() for x in re.findall(r'class="_0_DESC __sri-desc"[^>]*>(.*?)</div>', page, re.S)]
		i = 0
		for m in re.finditer(r'<a class="[^"]*__sri_title_link[^"]*"([^>]*)>(.*?)</a>', page, re.S):
			href_m = re.search(r'href="([^"]+)"', m.group(1))
			if not href_m:
				continue
			href = href_m.group(1)
			title_m = re.search(r'title="([^"]+)"', m.group(1))
			title = html.unescape(title_m.group(1) if title_m else re.sub(r"<[^>]+>", "", m.group(2))).strip()
			snip = descs[i] if i < len(descs) else ""
			i += 1
			rows.append({"t": 0, "url": href, "title": title, "snippet": snip, "time": "", "image": None, "props": {}, "group_id": _host_of(href), "paywalled": False})
	seen, out = set(), []
	for it in rows:
		u = (it.get("url") or "").strip()
		if not u or u in seen:
			continue
		seen.add(u)
		out.append(it)
	return out[:24], None


def parse_reverse_image_html(html):
	"""Kagi's reverse-image grid → rows carrying both the proxy and the source image."""
	soup = BeautifulSoup(html, "lxml")
	results = []
	for item in soup.find_all("div", class_="_0_image_item"):
		img_tag = item.find("img", class_="_0_img_src")
		link = item.find("a", class_="_0_img_link_el")
		proxy_src = ""
		if img_tag and img_tag.get("src"):
			src = img_tag["src"]
			proxy_src = src if src.startswith("http") else f"https://kagi.com{src}"
		results.append({"title": item.get("data-title", "").strip(), "url": item.get("data-host_url", "") or (link.get("href") if link else ""), "image_url": proxy_src, "original_url": item.get("data-content_url", proxy_src), "width": item.get("data-width"), "height": item.get("data-height")})
	return results


def reverse_image_search(image_bytes, filename="image.bin", mimetype="application/octet-stream", session=None):
	"""Upload one image to Kagi reverse search and return ``(payload, error)``. Errors come back as a string rather than an exception because the caller owns the HTTP status it turns into.
	Args:
		image_bytes: raw image body — callers own the file handle, not this function.
		filename: name sent in the multipart part; Kagi only uses it for the extension.
		mimetype: content type of the part.
		session: optional ``requests.Session`` so a caller can reuse its connection pool."""
	session_token = os.environ.get("KAGI_SESSION_TOKEN")
	if not session_token:
		return None, "KAGI_SESSION_TOKEN required for reverse image search"

	http = session or requests
	try:
		resp = http.post(REVERSE_IMAGE_URL, files={"file": (filename or "image.bin", image_bytes, mimetype or "application/octet-stream")}, cookies={"kagi_session": session_token}, headers={"User-Agent": REVERSE_IMAGE_UA, "Referer": "https://kagi.com/images"}, allow_redirects=True, timeout=60, verify=YUNA_TLS_VERIFY)
	except Exception as e:
		return None, str(e)

	if "login" in resp.url or "signin" in resp.url:
		return None, "Kagi session expired — refresh KAGI_SESSION_TOKEN"

	if "/images?" in resp.url and "reverse=upload" in resp.url:
		return {"meta": {"reverse_url": resp.url}, "data": parse_reverse_image_html(resp.content)}, None

	return None, "Unexpected response from Kagi reverse image upload"


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

		html = self._clean_html(html)
		meta = self._extract_metadata(html)
		title = meta.get("title", "")
		content_html = self._extract_content(html)
		plain = self._strip_tags(content_html or "")
		if len(plain.strip()) < 200:
			content_html = self._jsonld_body(html) or self._paragraphs(html)
			plain = self._strip_tags(content_html or "")

		if not content_html or len(plain.strip()) < 200:
			return title, ""

		if not title:
			title = self._extract_title_from_content(content_html)

		if output == "markdown":
			content = self._html_to_markdown(content_html, url)
		else:
			content = content_html

		text_only = self._strip_tags(content)
		if len(text_only.strip()) < 200:
			return "", ""

		return title.strip(), content.strip()

	def _download(self, url, timeout: int):
		"""Download HTML from URL."""
		try:
			headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
			response = requests.get(url, timeout=timeout, headers=headers, verify=YUNA_TLS_VERIFY)
			response.raise_for_status()
			if response.encoding == "ISO-8859-1":
				response.encoding = response.apparent_encoding
			return response.text
		except Exception:
			return ""

	def _clean_html(self, html):
		"""Remove unwanted elements from HTML."""
		html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
		html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
		html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
		html = re.sub(r"<noscript[^>]*>.*?</noscript>", "", html, flags=re.DOTALL | re.IGNORECASE)
		html = re.sub(r'<[^>]+style\s*=\s*["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>.*?</[^>]+>', "", html, flags=re.DOTALL | re.IGNORECASE)
		return html

	def _extract_metadata(self, html):
		"""Extract metadata from HTML."""
		meta = {}
		og = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']', html, re.I) or re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']', html, re.I)  # Prefer og:title (site chrome less often), then <title>. Only split on spaced separators so "COVID-19" / "state-of-the-art" survive.
		raw = ""
		if og:
			raw = og.group(1)
		else:
			title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
			if title_match:
				raw = title_match.group(1)
		if raw:
			title = re.sub(r"<[^>]+>", "", raw)
			title = self._decode_html_entities(title).strip()
			title = re.split(r"\s+[|\u2013\u2014]\s+|\s+-\s+", title, maxsplit=1)[0].strip()
			if title:
				meta["title"] = title
		return meta

	def _extract_content(self, html):
		"""Extract main article content. Nested divs used to win the first shallow match and drop the story."""
		best = ""
		best_len = 0
		for tag in ("article", "main"):
			for region in self._outer_regions(html, tag):
				text = self._strip_tags(region)
				if len(text) > best_len:
					best_len = len(text)
					best = region
		if best and best_len >= 200:
			return self._clean_content_block(best)
		blocks = self._split_into_blocks(html)
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

	def _outer_regions(self, html, tag):
		"""Outermost <tag>…</tag> regions, nesting counted so the article is not the first inner div."""
		pattern = re.compile(rf"</?{tag}\b[^>]*>", re.IGNORECASE)
		regions = []
		stack = []
		for match in pattern.finditer(html):
			token = match.group(0)
			if token.startswith("</"):
				if not stack:
					continue
				start = stack.pop()
				if not stack:
					regions.append(html[start:match.end()])
			elif token.endswith("/>"):
				continue
			else:
				stack.append(match.end())
		return regions

	def _jsonld_body(self, html):
		for match in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.I | re.S):
			try:
				data = json.loads(match.group(1))
			except Exception:
				continue
			body = self._ld_body(data)
			if body and len(body) >= 200:
				return body
		return ""

	def _ld_body(self, data):
		if isinstance(data, list):
			for item in data:
				found = self._ld_body(item)
				if found:
					return found
			return ""
		if not isinstance(data, dict):
			return ""
		body = data.get("articleBody") or data.get("text") or ""
		if isinstance(body, str) and len(body.strip()) >= 200:
			return body.strip()
		for key in ("@graph", "mainEntity"):
			found = self._ld_body(data.get(key))
			if found:
				return found
		return ""

	def _paragraphs(self, html):
		parts = []
		for match in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.I | re.S):
			text = self._strip_tags(match.group(1)).strip()
			if len(text) > 40:
				parts.append(text)
		return "\n\n".join(parts)

	def _split_into_blocks(self, html) -> list:
		"""Split HTML into content blocks."""
		containers = re.findall(r"<(div|article|section|main)[^>]*>(.*?)</\1>", html, re.IGNORECASE | re.DOTALL)
		blocks = [match[1] for match in containers]
		if not blocks:
			blocks = [html]
		return blocks

	def _score_block(self, block) -> float:
		"""Score a content block based on text density and content signals."""
		text = self._strip_tags(block)
		text_len = len(text.strip())
		if text_len < 100:
			return 0
		p_count = len(re.findall(r"<p[^>]*>", block, re.IGNORECASE))
		link_count = len(re.findall(r"<a[^>]*>", block, re.IGNORECASE))
		img_count = len(re.findall(r"<img[^>]*>", block, re.IGNORECASE))
		score = text_len
		score += p_count * 50  # Paragraphs are good
		score -= link_count * 20  # Too many links = navigation
		score += min(img_count * 30, 150)  # Images are good but cap it
		if re.search(r"(article|post|entry|content|main|story)", block, re.IGNORECASE):
			score += 100
		if re.search(r"(nav|menu|sidebar|footer|header|comment)", block, re.IGNORECASE):
			score -= 200
		return score

	def _clean_content_block(self, block):
		"""Clean extracted content block."""
		block = re.sub(r"<nav[^>]*>.*?</nav>", "", block, flags=re.DOTALL | re.IGNORECASE)
		block = re.sub(r"<aside[^>]*>.*?</aside>", "", block, flags=re.DOTALL | re.IGNORECASE)
		block = re.sub(r"<footer[^>]*>.*?</footer>", "", block, flags=re.DOTALL | re.IGNORECASE)
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
		for i in range(6, 0, -1):
			md = re.sub(f"<h{i}[^>]*>(.*?)</h{i}>", f"\n{'#' * i} \\1\n", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", md, flags=re.IGNORECASE | re.DOTALL)

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
		md = re.sub(r"<ul[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</ul>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<ol[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</ol>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<blockquote[^>]*>(.*?)</blockquote>", r"\n> \1\n", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", md, flags=re.IGNORECASE | re.DOTALL)
		md = re.sub(r"<p[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"</p>", "\n", md, flags=re.IGNORECASE)
		md = re.sub(r"<br[^>]*>", "\n", md, flags=re.IGNORECASE)
		md = self._strip_tags(md)
		md = re.sub(r"[ \t\xa0]{2,}", " ", md)  # collapse runs of spaces/tabs/nbsp
		md = re.sub(r"^[ \t\xa0]+|[ \t\xa0]+$", "", md, flags=re.MULTILINE)  # trim each line
		md = re.sub(r"\n{3,}", "\n\n", md)
		md = self._decode_html_entities(md)
		return md.strip()

	def _strip_tags(self, html):
		"""Remove all HTML tags."""
		return re.sub(r"<[^>]+>", "", html)

	def _decode_html_entities(self, text):
		"""Decode HTML entities (named + numeric)."""
		return html.unescape(text or "")


class Yuna_News:  # Yuna news — English intel stream (Kagi briefings + curated RSS). Locale boards (JP/KO/ZH/IT/RO/…) and sports/entertainment cats are cut at sync. Sync fans out every Kagi file + RSS feed in ONE parallel wave with Session reuse.
	KEEP_CATS = {"World", "USA", "Business", "Technology", "Science", "Gaming", "3D Printing", "AI", "Apple", "Canada", "Cryptocurrency", "Cybersecurity", "Google", "Linux & OSS", "Microsoft", "UK"}  # English tech / science / geopolitics only
	CAT_DENY = {"world cup", "wwe", "transfers", "cricket", "mlb", "football", "tennis", "mma", "aew", "slammiversary", "summerslam", "tournament", "injuries", "honours", "live music", "arts", "pop", "awards", "classic rock", "music rights", "charts", "tours", "album release", "new releases", "comedy", "disney", "superheroes", "casting", "obituaries", "obituary", "actor death", "martial arts", "franchise", "adaptations", "sci-fi", "crime", "mass shooting", "organized crime", "civil disorder", "traffic", "protest", "shark attack", "vatican", "sports betting", "soccer", "nfl", "nba", "nhl", "wrestling"}  # leftover English topic slop in World/USA/UK clusters
	RSS_NS = {"media": "http://search.yahoo.com/mrss/", "content": "http://purl.org/rss/1.0/modules/content/", "atom": "http://www.w3.org/2005/Atom", "dc": "http://purl.org/dc/elements/1.1/"}
	UA = "Mozilla/5.0 (compatible; YunaNews/2.0; +https://yunaai.com)"
	SYNC_WORKERS = 48
	CONNECT_TO = 3
	KAGI_READ_TO = 12
	RSS_READ_TO = 8
	KITE_TO = (3, 15)
	RSS_MAX_ITEMS = 20  # arXiv subcategory stays manageable; Apple feeds ~20–50

	def __init__(self, news_path=None):
		self.news_path = os.path.abspath(news_path or news_json_path())
		self._ensure_state()
		self.kagi_root = "https://news.kagi.com"
		self.kite_url = f"{self.kagi_root}/kite.json"
		self._tls = threading.local()
		self.rss_feeds = ["https://www.nature.com/nature.rss", "https://www.sciencedaily.com/rss/all.xml", "https://scitechdaily.com/feed/", "https://www.universetoday.com/feed/", "https://www.nasa.gov/rss/dyn/lg_image_of_the_day.rss", "https://www.nasa.gov/rss/dyn/breaking_news.rss", "https://rss.beehiiv.com/feeds/CHXHRVUx6h.xml", "https://rss.beehiiv.com/feeds/Vy37NcFo03.xml", "https://rss.beehiiv.com/feeds/h1qs4tUVIj.xml", "https://www.technologyreview.com/feed/", "https://www.quantamagazine.org/feed/", "https://feeds.arstechnica.com/arstechnica/technology-lab", "https://feeds.arstechnica.com/arstechnica/science", "https://feeds.arstechnica.com/arstechnica/apple", "https://9to5mac.com/feed/", "https://www.apple.com/newsroom/rss-feed.rss", "https://appleinsider.com/rss/news", "https://rss.arxiv.org/rss/cs.AI", "https://forum.betaprofiles.com/c/beta-software/5.rss", ]  # Curated English sources. Statista skipped (SSO). Full arXiv/cs too huge → cs.AI.

	def _bag(self):
		data = bag_load(self.news_path)
		data.setdefault("read", [])
		data.setdefault("news_bookmarks", [])
		data.setdefault("archive", {})
		if not isinstance(data["archive"], dict):
			data["archive"] = {}
		return data

	def _ensure_state(self):
		with _lock_for(self.news_path):  # Merge keys only — never bag_save a news-only object onto knowledge.json.
			data = bag_load(self.news_path)
			dirty = False
			if not isinstance(data.get("read"), list):
				data["read"] = []
				dirty = True
			if not isinstance(data.get("news_bookmarks"), list):
				data["news_bookmarks"] = []
				dirty = True
			if not isinstance(data.get("archive"), dict):
				data["archive"] = {}
				dirty = True
			if dirty:
				bag_save(self.news_path, data)

	def _get_state(self):
		"""Corrupt store raises — defaulting to empty drops every bookmark on the next save."""
		data = self._bag()
		return {"read": list(data.get("read") or []), "bookmarks": list(data.get("news_bookmarks") or [])}

	def _save_state(self, state):
		bag_update(self.news_path, read=list(state.get("read") or []), news_bookmarks=list(state.get("bookmarks") or []))

	def _archive_get(self, name):
		items = (self._bag().get("archive") or {}).get(name) or []
		return items if isinstance(items, list) else []

	def _archive_set(self, name, items):
		with _lock_for(self.news_path):
			data = self._bag()
			data.setdefault("archive", {})[name] = items
			bag_save(self.news_path, data)

	def _archive_del(self, name):
		with _lock_for(self.news_path):
			data = self._bag()
			(data.get("archive") or {}).pop(name, None)
			bag_save(self.news_path, data)

	def set_action(self, sid, action, value):
		if action not in ("read", "bookmarks"):
			return False
		state = self._get_state()
		state.setdefault("read", [])
		state.setdefault("bookmarks", [])
		if value and sid not in state[action]:
			state[action].append(sid)
		elif not value and sid in state[action]:
			state[action].remove(sid)
		self._save_state(state)
		return True

	@staticmethod
	def _script_bad(s):
		"""True if text is dominated by non-English scripts (CJK/Hangul/Cyrillic/Arabic/RO diacritics)."""
		s = s or ""
		n_cjk = n_hira = n_kata = n_hang = n_cyr = n_ar = n_lat = 0
		for ch in s:
			o = ord(ch)
			if 0x4E00 <= o <= 0x9FFF: n_cjk += 1
			elif 0x3040 <= o <= 0x309F: n_hira += 1
			elif 0x30A0 <= o <= 0x30FF: n_kata += 1
			elif 0xAC00 <= o <= 0xD7AF: n_hang += 1
			elif 0x0400 <= o <= 0x04FF: n_cyr += 1
			elif 0x0600 <= o <= 0x06FF: n_ar += 1
			elif ch.isalpha() and o < 128: n_lat += 1
		if n_hira + n_kata + n_hang + n_cyr + n_ar:
			return True
		if n_cjk >= 2 and n_cjk > n_lat:
			return True
		if any(ch in "șțȘȚ" for ch in s):  # Romanian-specific (not French â/î)
			return True
		return False

	def is_slop(self, item):
		"""Drop non-English + sports/entertainment residue. RSS is already curated."""
		title = item.get("title") or ""
		cat = (item.get("category") or "").strip()
		snip = item.get("snippet") or item.get("short_summary") or ""
		if item.get("_type") == "rss":
			return self._script_bad(title) or self._script_bad(snip[:400])
		blob = title + " " + snip
		if self._script_bad(title) or self._script_bad(cat) or self._script_bad(blob[:400]):
			return True
		if any(ord(c) >= 128 and c.isalpha() for c in cat):  # 外交 / 인공지능 / Infrastructură
			return True
		if cat.lower() in self.CAT_DENY:
			return True
		return False

	def _make_session(self):
		s = requests.Session()
		a = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=0)
		s.mount("https://", a)
		s.mount("http://", a)
		s.verify = YUNA_TLS_VERIFY
		s.headers["User-Agent"] = self.UA
		s.headers["Accept"] = "application/rss+xml, application/atom+xml, application/xml, application/json, text/xml, */*"
		return s

	def _session(self):
		s = getattr(self._tls, "s", None)
		if s is None:
			s = self._make_session()
			self._tls.s = s
		return s

	@staticmethod
	def _read_json(path, default):
		"""Missing archive file is normal; a corrupt one raises rather than reading as empty and letting the merge that follows write the survivors away."""
		if not os.path.exists(path):
			return default
		with open(path, "r") as f:
			return json.load(f)

	@staticmethod
	def _source_label(url):
		host = urlparse(url).netloc.replace("www.", "").lower()
		pretty = {"feeds.arstechnica.com": "Ars Technica", "rss.beehiiv.com": "Beehiiv", "rss.arxiv.org": "arXiv", "forum.betaprofiles.com": "BetaProfiles", "9to5mac.com": "9to5Mac", "appleinsider.com": "AppleInsider", "www.apple.com": "Apple Newsroom", "apple.com": "Apple Newsroom", "www.nature.com": "Nature", "www.sciencedaily.com": "ScienceDaily", "scitechdaily.com": "SciTechDaily", "www.universetoday.com": "Universe Today", "www.nasa.gov": "NASA", "www.technologyreview.com": "MIT Tech Review", "www.quantamagazine.org": "Quanta"}
		if host in pretty:
			return pretty[host]
		return host.split(".")[0].upper()

	def sync_daily(self):
		"""One parallel wave: every Kagi cat + every RSS feed, Session-pooled, fail-soft."""
		stats = {"yuna": 0, "rss": 0, "fail": 0, "skipped": 0}
		state = self._get_state()
		keep_ids = set(state["read"]) | set(state["bookmarks"])
		t0 = time.time()

		try:
			kite = self._make_session().get(self.kite_url, timeout=self.KITE_TO).json()
		except Exception as e:
			return {"error": f"kite fetch failed: {e}"}

		cats = [c for c in kite.get("categories", []) if c.get("name") in self.KEEP_CATS]
		jobs = [("kagi", c) for c in cats] + [("rss", u) for u in self.rss_feeds]
		total = len(jobs)
		kagi_raw = {}  # name → list of cluster dicts
		rss_ok = {}  # feed_url → [(sid, item), …] only successful fetches
		failed = []

		def _run(kind, payload):
			if kind == "kagi":
				return self._fetch_kagi_category(payload)
			return self._fetch_rss_feed(payload)

		workers = min(self.SYNC_WORKERS, max(8, total))
		print(f"[YunaNews] sync start — {len(cats)} kagi + {len(self.rss_feeds)} rss = {total} jobs, workers={workers}")
		with ThreadPoolExecutor(max_workers=workers) as pool:
			futs = {pool.submit(_run, kind, payload): (kind, payload) for kind, payload in jobs}
			for i, fut in enumerate(as_completed(futs), 1):
				kind, payload = futs[fut]
				label = payload.get("name") if kind == "kagi" else payload
				try:
					out = fut.result()
				except Exception as e:
					stats["fail"] += 1
					failed.append(f"{kind}:{label}")
					print(f"[{i}/{total}] FAIL {kind}:{label} — {e}")
					continue
				if kind == "kagi":
					name, clusters, err = out
					if err:
						stats["fail"] += 1
						failed.append(f"kagi:{name}")
						print(f"[{i}/{total}] FAIL kagi:{name} — {err}")
					else:
						kagi_raw[name] = clusters
						print(f"[{i}/{total}] OK   kagi:{name} ({len(clusters)} clusters)")
				else:
					url, pairs, err = out
					if err:
						stats["fail"] += 1
						failed.append(f"rss:{url}")
						print(f"[{i}/{total}] FAIL rss:{url} — {err}")
					else:
						rss_ok[url] = pairs
						print(f"[{i}/{total}] OK   rss:{self._source_label(url)} (+{len(pairs)})")

		for name, clusters in kagi_raw.items():  # Serial disk merge — no concurrent JSON write races.
			added, skipped = self._merge_kagi_file(name, clusters, keep_ids)
			stats["yuna"] += added
			stats["skipped"] += skipped
		added_rss, skipped_rss = self._merge_rss_stream(rss_ok, keep_ids)
		stats["rss"] += added_rss
		stats["skipped"] += skipped_rss
		self._prune_stale_kagi_files(keep_ids)  # Prune stale locale archive files left from older syncs (Japan.json etc.).

		elapsed = time.time() - t0
		print(f"[YunaNews] sync done: +{stats['yuna']} yuna, +{stats['rss']} rss, {stats['skipped']} slop skipped, {stats['fail']} failed in {elapsed:.1f}s")
		out = {"status": "success", "stats": stats, "elapsed": round(elapsed, 2)}
		if failed:
			out["failed"] = failed
		return out

	def _fetch_kagi_category(self, cat):
		"""Network-only. Return (name, clusters, err)."""
		name = cat.get("name") or "?"
		try:
			r = self._session().get(f"{self.kagi_root}/{cat['file']}", timeout=(self.CONNECT_TO, self.KAGI_READ_TO))
			r.raise_for_status()
			data = r.json()
			return name, data.get("clusters") or [], None
		except Exception as e:
			return name, [], str(e)

	def _fetch_rss_feed(self, url):
		"""Network-only. Return (url, [(sid, item), …], err)."""
		ns = self.RSS_NS
		try:
			r = self._session().get(url, timeout=(self.CONNECT_TO, self.RSS_READ_TO))
			r.raise_for_status()
			raw = r.content.replace(b"\x00", b"")
			try:
				root = ET.fromstring(raw)
			except ET.ParseError:
				soup = BeautifulSoup(raw, "xml")  # beehiiv ships illegal XML tokens
				root = ET.fromstring(str(soup).encode("utf-8"))
		except Exception as e:
			return url, [], str(e)

		items = root.findall(".//item") or root.findall(".//{http://purl.org/rss/1.0/}item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")  # RSS 2.0 <item>, Atom <entry>, RSS 1.0 RDF <item> (Nature).
		source = self._source_label(url)
		out = []
		for i in items[:self.RSS_MAX_ITEMS]:
			try:
				title = (i.findtext("title") or i.findtext("{http://purl.org/rss/1.0/}title") or i.findtext("{http://www.w3.org/2005/Atom}title") or "Untitled")
				title = html.unescape(title.strip()) if isinstance(title, str) else "Untitled"
				link = i.findtext("link") or i.findtext("{http://purl.org/rss/1.0/}link")
				if not link:
					for link_node in i.findall("{http://www.w3.org/2005/Atom}link") or i.findall("atom:link", ns):  # Prefer Atom rel=alternate (or bare link); skip rel=self/related.
						rel = (link_node.get("rel") or "alternate").lower()
						href = link_node.get("href")
						if href and rel in ("alternate", ""):
							link = href
							break
					if not link:
						for link_node in i.findall("{http://www.w3.org/2005/Atom}link") or i.findall("atom:link", ns):
							href = link_node.get("href")
							if href:
								link = href
								break
					if not link:
						link = i.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about")
				if not link or not isinstance(link, str):
					continue
				sid = f"rss_{hashlib.md5(link.encode()).hexdigest()}"
				desc = i.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or i.findtext("description") or i.findtext("{http://purl.org/rss/1.0/}description") or i.findtext("{http://www.w3.org/2005/Atom}summary") or i.findtext("{http://www.w3.org/2005/Atom}content") or ""  # NEVER pass ns dict as findtext's 2nd positional arg — that's `default`, not namespaces.
				if not isinstance(desc, str):
					desc = ""
				snippet = html.unescape(re.sub(r"<[^>]+>", " ", desc))
				snippet = re.sub(r"\s+", " ", snippet).strip()
				snippet = re.sub(r"(?i)\s*Credit:\s*.*$", "", snippet).strip()  # NASA/etc. photo credits after the blurb
				snippet = snippet[:350]
				if snippet and not snippet.endswith("..."):
					snippet += "..."
				body = html.unescape(re.sub(r"<[^>]+>", " ", desc))
				body = re.sub(r"\s+", " ", body).strip()[:20000]
				thumb = None
				mc = i.find("{http://search.yahoo.com/mrss/}content")
				if mc is None:
					mc = i.find("media:content", ns)
				if mc is not None:
					thumb = mc.get("url")
				if not thumb:
					enc = i.find("enclosure")
					if enc is not None:
						thumb = enc.get("url")
				if not thumb and desc:
					img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', desc)
					if img_match:
						thumb = img_match.group(1)
				pubd = i.findtext("pubDate") or i.findtext("{http://purl.org/dc/elements/1.1/}date") or i.findtext("{http://purl.org/rss/1.0/}date") or i.findtext("{http://www.w3.org/2005/Atom}updated") or i.findtext("{http://www.w3.org/2005/Atom}published")
				stamp = time.time()
				if pubd and isinstance(pubd, str):
					try:
						dt = email.utils.parsedate_to_datetime(pubd)
						if dt is not None:
							stamp = dt.timestamp()
						else:
							raise ValueError("unparsed")
					except Exception:
						try:
							stamp = datetime.datetime.fromisoformat(pubd.replace("Z", "+00:00")).timestamp()
						except Exception:
							pass
				item = {"_id": sid, "_type": "rss", "source": source, "title": title, "url": link, "snippet": snippet, "body": body, "timestamp": stamp, "image_url": thumb, "feed_url": url}
				self._stamp_app_url(item)
				if not self.is_slop(item):
					out.append((sid, item))
			except Exception as e:
				print(f"  rss item skip ({url}): {e}")
				continue
		return url, out, None

	def _merge_kagi_file(self, name, clusters, keep_ids):
		"""Serial merge of one Kagi category. Returns (added, skipped)."""
		key = name.lower().replace(" ", "_").replace("&", "and")
		existing_map = {x["_id"]: x for x in self._archive_get(key)}
		fresh_ids = set()
		added = skipped = 0
		for c in clusters:
			title = (c.get("title") or "").strip()
			if not title:
				continue
			ts = c.get("timestamp")
			try:
				ts = int(ts) if ts is not None else 0
			except (TypeError, ValueError):
				ts = 0
			sid = f"kagi_{ts}_{hashlib.md5(title.encode()).hexdigest()[:8]}"
			img_u = c.get("primary_image", "")
			if isinstance(img_u, dict):
				img_u = img_u.get("url", "")
			item = {"_id": sid, "_type": "yuna", "title": title, "short_summary": c.get("short_summary"), "timestamp": ts or None, "category": c.get("category"), "talking_points": c.get("talking_points") or [], "scientific_significance": c.get("scientific_significance") or [], "perspectives": c.get("perspectives") or [], "unique_domains": c.get("unique_domains") or 1, "did_you_know": c.get("did_you_know") or "", "historical_background": c.get("historical_background") or "", "technical_details": c.get("technical_details") or [], "timeline": c.get("timeline") or [], "image_url": img_u or None, "url": f"https://kagi.com/news/search?q={quote(title)}", "kagi_file": name}
			self._stamp_app_url(item)
			if self.is_slop(item):
				skipped += 1
				continue
			fresh_ids.add(sid)
			if sid not in existing_map:
				existing_map[sid] = item
				added += 1
			else:
				existing_map[sid] = item  # refresh fields
		final = [item for sid, item in existing_map.items() if sid in fresh_ids or sid in keep_ids]  # keep_ids wins over is_slop so bookmarks/read never get wiped from disk
		final.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
		self._archive_set(key, final)
		return added, skipped

	def _merge_rss_stream(self, rss_ok, keep_ids):
		"""Merge successful feeds only. Failed feeds keep their prior rows (fail-soft)."""
		if not rss_ok:
			print("[YunaNews] RSS merge skipped — zero successful feeds (keeping personal_stream)")
			return 0, 0
		existing = {x["_id"]: x for x in self._archive_get("personal_stream")}
		ok_feeds = set(rss_ok)
		fresh = set()
		added = skipped = 0
		for url, pairs in rss_ok.items():
			for sid, item in pairs:
				if self.is_slop(item):
					skipped += 1
					continue
				fresh.add(sid)
				if sid not in existing:
					existing[sid] = item
					added += 1
				else:
					existing[sid] = item
		final = []  # Keep: this run's fresh items, bookmarks/read, OR rows from feeds that failed this sync.
		for sid, item in existing.items():
			feed = item.get("feed_url")
			if sid in fresh or sid in keep_ids or (feed and feed not in ok_feeds):
				final.append(item)
		final.sort(key=lambda x: (x.get("timestamp") or 0), reverse=True)
		self._archive_set("personal_stream", final)
		return added, skipped

	def _prune_stale_kagi_files(self, keep_ids):
		"""Drop archive keys for locale boards we no longer sync (unless bookmarked/read)."""
		keep = {n.lower().replace(" ", "_").replace("&", "and") for n in self.KEEP_CATS}
		keep.add("personal_stream")
		archive = dict((self._bag().get("archive") or {}))
		for name, items in list(archive.items()):
			if name in keep:
				continue
			kept = [i for i in (items or []) if i.get("_id") in keep_ids]
			if kept:
				self._archive_set(name, kept)
			else:
				self._archive_del(name)
				print(f"[YunaNews] pruned stale archive {name}")

	def _iter_archive_files(self):
		archive = self._bag().get("archive") or {}
		for name, items in archive.items():
			yield f"{name}.json", (name == "personal_stream"), items if isinstance(items, list) else []

	@staticmethod
	def app_url_for(sid):
		"""Deep link into Yuna Ai SwiftUI News Studio (`yunaai://news?id=`)."""
		if not sid:
			return ""
		return f"yunaai://news?id={quote(str(sid), safe='')}"

	@classmethod
	def _stamp_app_url(cls, item):
		sid = item.get("_id")
		if sid:
			item["app_url"] = cls.app_url_for(sid)
		return item

	def get_by_id(self, sid):
		"""Single archive row by `_id`, or None. Stamps `app_url`."""
		if not sid:
			return None
		state = self._get_state()
		read, bookmarks = set(state["read"]), set(state["bookmarks"])
		for _f, _is_rss, items in self._iter_archive_files():
			for s in items:
				if s.get("_id") != sid:
					continue
				row = dict(s)
				row["_read"] = sid in read
				row["_bookmarked"] = sid in bookmarks
				return self._stamp_app_url(row)
		return None

	def get_feed(self, q=None, cat=None, yuna_only=False, whole=False):
		"""Feed is unread. A query searches the whole archive, including rows already read."""
		state = self._get_state()
		read, bookmarks = set(state["read"]), set(state["bookmarks"])
		clean_cat = cat.lower().replace(" ", "_").replace("&", "and") if cat else None
		ql = q.lower() if q else None
		results = []
		seen = set()  # same Kagi story lands in multiple category files → one row

		for f, is_rss, items in self._iter_archive_files():
			if yuna_only and is_rss:
				continue
			if clean_cat and not is_rss and clean_cat not in f:
				continue
			for s in items:
				sid = s.get("_id")
				if not sid or sid in seen or (sid in read and not whole):
					continue
				if self.is_slop(s):
					continue
				content = (str(s.get("title", "")) + " " + str(s.get("short_summary", "")) + " " + str(s.get("snippet", "")) + " " + str(s.get("body", ""))).lower()
				rank = self._query_rank(ql, s) if ql else 0
				if ql and rank is None:
					continue
				if clean_cat and is_rss:
					cat_keyword = cat.lower().split()[0]  # Prefer source/feed match for Apple; else keyword on title/snippet.
					src = str(s.get("source", "")).lower()
					feed = str(s.get("feed_url", "")).lower()
					appleish = cat_keyword == "apple" and ("apple" in src or "9to5" in src or "apple" in feed or "9to5" in feed)
					if not appleish and cat_keyword not in content and cat_keyword not in src:
						continue
				s = dict(s)
				s["_bookmarked"] = sid in bookmarks
				s["_rank"] = rank if ql else 0
				self._stamp_app_url(s)
				seen.add(sid)
				results.append(s)

		def _sort_key(x):  # Yuna briefings often have null timestamps — surface them ahead of sinking to the bottom.
			ts = x.get("timestamp") or 0
			boost = 1 if x.get("_type") == "yuna" and not ts else 0
			return (x.get("_rank") or 0, -boost, -(ts or 0))

		results.sort(key=_sort_key)
		return results

	@staticmethod
	def _query_rank(q, item):
		"""0 if every word is in the title, 1 if it is only in the summary or stored body."""
		if not q:
			return 0
		tokens = [part for part in q.lower().split() if part]
		if not tokens:
			return 0
		title = str(item.get("title") or "").lower()
		body = " ".join(str(item.get(key) or "") for key in ("short_summary", "snippet", "body")).lower()

		def hit(text):
			folded = text.replace(" ", "")
			query = "".join(tokens)
			return all(token in text for token in tokens) or (query and query in folded)

		if hit(title):
			return 0
		if hit(body):
			return 1
		return None

	def stored_body(self, url):
		if not url:
			return "", ""
		for _f, _is_rss, items in self._iter_archive_files():
			for item in items:
				if item.get("url") == url or item.get("original_url") == url:
					body = item.get("body") or ""
					if len(body) >= 200:
						return item.get("title") or "", body
		return "", ""

	def get_saved(self):
		state = self._get_state()
		read, bookmarks = set(state["read"]), set(state["bookmarks"])
		all_data = []
		seen = set()  # same Kagi story lands in several category files → one row
		for _f, _is_rss, items in self._iter_archive_files():
			for s in items:
				sid = s.get("_id")
				if not sid or sid in seen:
					continue
				kept = sid in bookmarks or sid in read
				if not kept and self.is_slop(s):
					continue  # hide slop in archives only when not user-saved
				seen.add(sid)
				row = dict(s)
				row["_read"] = sid in read
				row["_bookmarked"] = sid in bookmarks
				self._stamp_app_url(row)
				all_data.append(row)
		return {"bookmarks": [s for s in all_data if s["_bookmarked"]], "history": [s for s in all_data if s["_read"]]}

	def get_world_news(self, query):
		key = os.environ.get("API_KAGI_KEY")
		if not key:
			return []
		try:
			headers = {"Authorization": f"Bot {key}"}
			r = self._make_session().get("https://kagi.com/api/v0/enrich/news", params={"q": query}, headers=headers, timeout=10)
			data = r.json().get("data") or []
			return [d for d in data if not self._script_bad((d.get("title") or "") + " " + (d.get("snippet") or ""))]
		except Exception:
			return []


def _ics_field(text, name):
	for line in (text or "").splitlines():
		if line.startswith(name + ":") or line.startswith(name + ";"):
			return line.split(":", 1)[-1].strip()
	return ""


class Yuna_Calendar:  # Yuna calendar — requests, not the caldav package. That client stalls on iCloud's calendar-home PROPFIND (niquests HTTP/2).
	def __init__(self):
		self.user = get_env("EMAIL_YUKI")
		self.password = get_env("APPLE_APP_PASSWORD_YUKI")
		self.url = "https://caldav.icloud.com/"
		self.tz = ZoneInfo("America/Edmonton")
		self._auth = (self.user, self.password)
		self._ns = {"d": "DAV:", "c": "urn:ietf:params:xml:ns:caldav"}

	def _abs(self, href):
		if href.startswith("http://") or href.startswith("https://"):
			return href
		return urljoin(self.url, href)

	def _propfind(self, url, body, depth="0"):
		r = requests.request("PROPFIND", url, data=body.encode(), headers={"Depth": depth, "Content-Type": "application/xml"}, auth=self._auth, timeout=15)
		r.raise_for_status()
		return ET.fromstring(r.content)

	def _home(self):
		root = self._propfind(self.url, '<propfind xmlns="DAV:"><prop><current-user-principal/></prop></propfind>')
		principal = self._abs(root.find(".//d:current-user-principal/d:href", self._ns).text)
		home_xml = self._propfind(principal, '<propfind xmlns="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"><prop><c:calendar-home-set/></prop></propfind>')
		return self._abs(home_xml.find(".//c:calendar-home-set/d:href", self._ns).text)

	def calendars(self):
		listing = self._propfind(self._home(), '<propfind xmlns="DAV:"><prop><displayname/><resourcetype/></prop></propfind>', depth="1")
		out = []
		for resp in listing.findall("d:response", self._ns):
			href_el = resp.find("d:href", self._ns)
			rtype = resp.find(".//d:resourcetype", self._ns)
			kinds = [c.tag.split("}")[-1] for c in list(rtype)] if rtype is not None else []
			if href_el is None or "calendar" not in kinds:
				continue
			name_el = resp.find(".//d:displayname", self._ns)
			out.append(((name_el.text if name_el is not None and name_el.text else ""), self._abs(href_el.text)))
		return out

	def _pick(self, name=None):
		cals = [(n, u) for n, u in self.calendars() if "reminder" not in n.lower()]
		if not cals:
			raise RuntimeError("no calendars")
		want = (name or "Home").strip().lower()
		for n, url in cals:
			if n.strip().lower() == want:
				return n, url
		return cals[0]

	def get_events(self, start_date=None, end_date=None):
		if not start_date:
			start_date = datetime.datetime.now(self.tz)
		if not end_date:
			end_date = start_date + datetime.timedelta(days=1)
		if start_date.tzinfo is None:
			start_date = start_date.replace(tzinfo=self.tz)
		if end_date.tzinfo is None:
			end_date = end_date.replace(tzinfo=self.tz)
		start_utc = start_date.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
		end_utc = end_date.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
		body = f'<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav"><d:prop><c:calendar-data/></d:prop><c:filter><c:comp-filter name="VCALENDAR"><c:comp-filter name="VEVENT"><c:time-range start="{start_utc}" end="{end_utc}"/></c:comp-filter></c:comp-filter></c:filter></c:calendar-query>'
		results = []
		for name, url in self.calendars():
			if "reminder" in name.lower():
				continue
			r = requests.request("REPORT", url, data=body.encode(), headers={"Depth": "1", "Content-Type": "application/xml"}, auth=self._auth, timeout=20)
			if r.status_code not in (200, 207):
				continue
			root = ET.fromstring(r.content)
			for data in root.findall(".//c:calendar-data", self._ns):
				text = data.text or ""
				results.append({"title": _ics_field(text, "SUMMARY") or "(no title)", "start": _ics_field(text, "DTSTART"), "location": _ics_field(text, "LOCATION"), "calendar": name, "uid": _ics_field(text, "UID")})
		results.sort(key=lambda x: x["start"])
		return results

	def add_event(self, title, start, end=None, location="", calendar="Home"):
		if end is None:
			end = start + datetime.timedelta(hours=1)
		if start.tzinfo is None:
			start = start.replace(tzinfo=self.tz)
		if end.tzinfo is None:
			end = end.replace(tzinfo=self.tz)
		name, url = self._pick(calendar)
		uid = f"yuna-{uuid.uuid4().hex}@yunaai.com"
		stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
		ics = "\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Yuna//EN", "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{stamp}", f"DTSTART:{start.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", f"DTEND:{end.astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", f"SUMMARY:{title}", f"LOCATION:{location}", "END:VEVENT", "END:VCALENDAR", ""])
		target = url.rstrip("/") + "/" + uid + ".ics"
		r = requests.put(target, data=ics.encode(), headers={"Content-Type": "text/calendar; charset=utf-8"}, auth=self._auth, timeout=20)
		if r.status_code not in (201, 204):
			raise RuntimeError(f"calendar PUT {r.status_code} {r.text[:180]}")
		return {"title": title, "start": start.isoformat(), "end": end.isoformat(), "calendar": name, "uid": uid, "href": target}


def _p8_path():
	"""WeatherKit key. Env path wins when the file is there; otherwise the Setup copy of the same AuthKey name."""
	path = get_env("APPLE_PRIVATE_KEY_PATH") or ""
	if path and os.path.isfile(path):
		return path
	name = os.path.basename(path) if path else "AuthKey_G7F3HA7R4S.p8"
	for root in (os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Personal/Github/yuna-ai/lib/Yuna Small Apps/Yuna Setup"), os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/Personal/WebCert")):
		cand = os.path.join(root, name)
		if os.path.isfile(cand):
			return cand
	return path


class Yuna_Weather:
	def __init__(self):
		self.team_id = get_env("APPLE_TEAM_ID")
		self.key_id = get_env("APPLE_KEY_ID")
		self.client_id = get_env("APPLE_CLIENT_ID", "com.yuna-ai.app")  # WeatherKit is enabled on this service id. com.yunaai.app returns NOT_ENABLED.
		self.private_key_path = _p8_path()
		if not self.key_id and self.private_key_path and os.path.basename(self.private_key_path).startswith("AuthKey_"):
			self.key_id = os.path.basename(self.private_key_path)[len("AuthKey_"):-len(".p8")]

	def _get_token(self, scope="weather"):
		if not self.private_key_path or not os.path.isfile(self.private_key_path):
			return None
		with open(self.private_key_path, "r") as f:
			pk = f.read()

		now = int(time.time())
		payload = {"iss": self.team_id, "iat": now, "exp": now + 3600}
		if scope in ("weather", "map"):
			payload["sub"] = self.client_id

		return jwt.encode(payload=payload, key=pk, algorithm="ES256", headers={"kid": self.key_id, "id": f"{self.team_id}.{self.client_id}"} if scope == "weather" else {"kid": self.key_id, "typ": "JWT"})

	def get_weather_at(self, lat, lon):
		w_token = self._get_token("weather")
		if not w_token:
			return {"error": "no weatherkit key"}
		w_url = f"https://weatherkit.apple.com/api/v1/weather/en/{lat}/{lon}?dataSets=currentWeather,forecastDaily"
		w_resp = requests.get(w_url, headers={"Authorization": f"Bearer {w_token}"}, timeout=(5, 15), verify=YUNA_TLS_VERIFY)
		if w_resp.status_code != 200:
			return {"error": f"WeatherKit {w_resp.status_code}", "body": w_resp.text[:180]}
		return w_resp.json()

	def get_weather(self, city="Calgary"):
		map_token = self._get_token("map")  # Geocode (MapKit)
		geo_url = f"https://maps-api.apple.com/v1/geocode?q={quote(city)}&lang=en-US"
		geo_resp = requests.get(geo_url, headers={"Authorization": f"Bearer {map_token}"}, timeout=(5, 15), verify=YUNA_TLS_VERIFY)
		if geo_resp.status_code != 200:
			return {"error": "Geocode failed"}
		coords = geo_resp.json()["results"][0]["coordinate"]
		lat, lon = coords["latitude"], coords["longitude"]
		w_token = self._get_token("weather")  # Weather (WeatherKit)
		w_url = f"https://weatherkit.apple.com/api/v1/weather/en/{lat}/{lon}?dataSets=currentWeather,forecastDaily"
		w_resp = requests.get(w_url, headers={"Authorization": f"Bearer {w_token}"}, timeout=(5, 15), verify=YUNA_TLS_VERIFY)
		return w_resp.json()


def _parse_skill_lines(body):
	"""Parse ``Name -> args`` lines from a GreatSage(...) body."""
	body = (body or "").strip()
	if not body:
		return []
	out = []
	for raw in body.splitlines():
		m = SKILL_LINE_RE.match(raw)
		if not m:
			continue
		out.append((m.group("name").strip().lower(), m.group("args").strip()))
	if out:
		return out
	m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.*)\s*$", body, re.DOTALL)  # Single-line body without newlines: Search -> "query"
	if m:
		return [(m.group(1).strip().lower(), m.group(2).strip())]
	return []


def _iter_action_bodies(text):
	"""Yield ``(wrapper, body)`` for every ``<action>Name(…)`` block."""
	for m in ACTION_OPEN_RE.finditer(text):
		close = ACTION_CLOSE_RE.search(text, m.end())
		if close:
			body = text[m.end():close.start()]
		else:
			nxt = ACTION_UNCLOSED_END_RE.search(text, m.end())
			body = text[m.end():nxt.start() if nxt else len(text)]
		body = body.rstrip()
		if body.endswith(")") and body.count(")") > body.count("("):
			body = body[:-1].rstrip()  # that ")" closes Name(; a balanced one belongs to the last argument
		yield (m.group("wrap") or "GreatSage"), body


def _parse_colon_skills(body):
	"""``wikipedia: Nelumbo`` when she skips the arrow. Only known skill names."""
	known = {"search", "searchwikipedia", "wikipedia", "wikipediasearch", "checkweather", "weather", "setcalendar", "calendar", "getcalendar", "dictionary", "dict", "lookup", "define", "summarize", "summary", "readpage", "getnewsheadlines", "news", "convertcurrency", "currency", "calculatemath", "math", "settimer", "timer", "setreminder", "reminder", "setalarm", "alarm", "gettime", "time", "now"}
	out = []
	for raw in (body or "").splitlines():
		if "->" in raw:
			continue
		m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$", raw)
		if not m or m.group(1).strip().lower() not in known:
			continue
		out.append((m.group(1).strip().lower(), m.group(2).strip()))
	return out


def parse_actions(text):
	"""Return ``[(name_lower, args_str), …]`` from every action block in ``text``."""
	if not text:
		return []
	out = []
	for wrap, body in _iter_action_bodies(text):
		lines = _parse_skill_lines(body) or _parse_colon_skills(body)
		if lines:
			out.extend(lines)
			continue
		wrap_l = (wrap or "").strip().lower()
		blob = (body or "").strip()
		if not blob:
			continue
		if wrap_l in ("wikipedia", "searchwikipedia", "wikipediasearch"):
			out.append(("summarize" if "http://" in blob or "https://" in blob else "searchwikipedia", blob))
		elif wrap_l not in ("greatsage", "action"):
			out.append((wrap_l, blob))
	if out:
		return out
	head = text  # Naked continuations sometimes emit the skill line and forget the GreatSage wrapper. Take only the lines before the next tag so a later catalog dump is not executed.
	for stop in ("<yuki>", "<yuna>", "<data>", "<aibo>", "</yuna>", "<action>"):
		i = head.find(stop)
		if i >= 0:
			head = head[:i]
	return _parse_skill_lines(head)


def close_action_tag(text):
	"""Re-append ``</action>`` when generation stopped on that token."""
	text = text or ""
	if ACTION_CLOSE_RE.search(text) or not parse_actions(text):
		return text
	return text.rstrip() + "</action>"


def _strip_quotes(s):
	s = (s or "").strip()
	if len(s) >= 2 and s[0] in "\"'“”‘’" and s[-1] in "\"'“”‘’":
		return s[1:-1].strip()
	return s


def _parse_kv(args):
	"""Parse ``key: value, key2: value2`` — commas inside quotes are kept."""
	args = (args or "").strip()
	if not args:
		return {}
	out = {}
	parts, buf, q = [], "", None  # Split on commas that aren't inside quotes.
	for ch in args:
		if ch in "\"'“”‘’":
			if q is None:
				q = ch
			elif ch == q or (q in "\"'" and ch in "\"'") or (q in "“”" and ch in "“”"):
				q = None
			buf += ch
		elif ch == "," and q is None:
			parts.append(buf)
			buf = ""
		else:
			buf += ch
	if buf.strip():
		parts.append(buf)
	for p in parts:
		if ":" not in p:
			continue
		k, v = p.split(":", 1)
		out[k.strip().lower()] = _strip_quotes(v)
	if not out and args:
		out["query"] = _strip_quotes(args)  # Bare quoted search string with no keys → treat as query
	return out


def dispatch_actions(actions, char_limit=1000):
	"""Run every ``(name, args)`` and join snippets into one ``<data>`` body."""
	actions = actions or []
	if not actions:
		return "(no Extra Skills to run)"
	per = max(180, char_limit // max(1, len(actions)))
	parts = []
	budget = char_limit
	for name, args in actions:
		if budget <= 32:
			break
		snip = dispatch_action(name, args, char_limit=min(per, budget))
		parts.append(snip)
		budget -= len(snip) + 1
	return "\n".join(parts)


def dispatch_action(name, args, char_limit=1000):
	"""Run one Extra Skill; always returns a string (never raises)."""
	name = (name or "").strip().lower().replace("_", "")
	print(f"[GreatSage] dispatch: name={name!r}  args={args!r}")
	kv = _parse_kv(args)
	try:
		if name in ("search", ):
			q = kv.get("query") or _strip_quotes(args)
			return _action_search(q, char_limit=char_limit)
		if name in ("searchwikipedia", "wikipedia", "wikipediasearch", "wikisearch"):
			if kv.get("url") or "http://" in (args or "") or "https://" in (args or ""):
				return _cap_text(_action_summarize(kv if kv.get("url") else {"url": next((p for p in args.split() if p.startswith("http")), "")}), char_limit)
			return _cap_text(_action_wikipedia(kv.get("query") or kv.get("topic") or kv.get("q") or kv.get("search") or _strip_quotes(args)), char_limit)
		if name in ("checkweather", "weather"):
			return _cap_text(_action_weather(kv), char_limit)
		if name in ("setcalendar", "calendar", "addevent"):
			return _cap_text(_action_calendar_set(kv), char_limit)
		if name in ("getcalendar", "listevents"):
			return _cap_text(_action_calendar_get(kv), char_limit)
		if name in ("convertcurrency", "currency"):
			return _cap_text(_action_currency(kv), char_limit)
		if name in ("settimer", "timer"):
			return _cap_text(_action_timer(kv), char_limit)
		if name in ("setreminder", "reminder"):
			return _cap_text(_action_reminder(kv), char_limit)
		if name in ("setalarm", "alarm"):
			return _cap_text(_action_alarm(kv), char_limit)
		if name in ("calculatemath", "math", "calculate"):
			return _cap_text(_action_math(kv.get("expression") or _strip_quotes(args)), char_limit)
		if name in ("translatetext", "translate"):
			return _cap_text(_action_translate(kv), char_limit)
		if name in ("getnewsheadlines", "news", "getnews"):
			return _cap_text(_action_news(kv), char_limit)
		if name in ("dictionary", "dict", "lookup", "define"):
			return _cap_text(_action_dictionary(kv), char_limit)
		if name in ("summarize", "summary", "readpage"):
			return _cap_text(_action_summarize(kv), char_limit)
		if name in ("gettime", "time", "now"):
			return _cap_text(_action_time(kv), char_limit)
		if name in ("sendtext", "text"):
			return _cap_text(_action_send_text(kv), char_limit)
		if name in ("orderpizza", "pizza"):
			return _cap_text(_action_pizza(kv), char_limit)
	except Exception as e:
		print(f"[GreatSage] {name} failed: {e}")
		traceback.print_exc()
		return f"({name}: error — {e})"
	msg = f"(GreatSage: Extra Skill {name!r} is not implemented yet)"
	print(f"[GreatSage] {msg}")
	return msg


def _wiki_summary(title):
	r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}", timeout=10, headers={"User-Agent": "Yuna/GreatSage"}, verify=YUNA_TLS_VERIFY)
	if r.status_code != 200:
		return ""
	j = r.json()
	extract = (j.get("extract") or "").strip()
	name = (j.get("title") or title).strip()
	return f"{name}: {extract}" if extract else ""


def _action_wikipedia(query):
	q = (query or "").strip()
	if ":" in q and q.split(":", 1)[0].strip().lower() in ("topic", "query", "q"):
		q = q.split(":", 1)[1].strip()
	for suffix in (" biography", " bio", " wikipedia"):
		if q.lower().endswith(suffix):
			q = q[:-len(suffix)].strip()
	if not q:
		return "(empty wikipedia query)"
	hit = _wiki_summary(q)
	if hit and "disambiguation" not in hit.lower()[:80]:
		return hit
	r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "opensearch", "search": q, "limit": 5, "namespace": 0, "format": "json"}, timeout=10, headers={"User-Agent": "Yuna/GreatSage"}, verify=YUNA_TLS_VERIFY)
	payload = r.json()
	if len(payload) > 1 and payload[1]:
		titles = [t for t in payload[1] if "disambiguation" not in t.lower() and "biography" not in t.lower()]
		title = titles[0] if titles else payload[1][0]
		blurb = payload[2][0] if len(payload) > 2 and payload[2] else ""
		return _wiki_summary(title) or f"{title}: {blurb}".strip()
	return f"(wikipedia: no page for {q})"


def _resolve_day(date_s, tz):
	now = datetime.datetime.now(tz).date()
	key = (date_s or "today").strip().lower()
	if key in ("today", "now", ""):
		return now
	if key == "tomorrow":
		return now + datetime.timedelta(days=1)
	week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
	short = [w[:3] for w in week]
	if key in week or key[:3] in short:
		target = week.index(key) if key in week else short.index(key[:3])
		return now + datetime.timedelta(days=(target - now.weekday()) % 7)
	return datetime.date.fromisoformat(key[:10])


def _resolve_local(date_s, time_s, tz=None):
	tz = tz or ZoneInfo("America/Edmonton")
	day = _resolve_day(date_s, tz)
	hh, mm = 9, 0
	if time_s and str(time_s).strip():
		parts = str(time_s).strip().split(":")
		hh = int(parts[0])
		mm = int(parts[1]) if len(parts) > 1 else 0
	return datetime.datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)


def _action_calendar_set(kv):
	title = kv.get("title") or kv.get("event") or kv.get("summary") or ""
	if not title:
		return "calendar: need a title"
	if not (kv.get("date") or "").strip():
		return "calendar: need a date"
	start = _resolve_local(kv.get("date"), kv.get("time") or "09:00")
	calendar_name = kv.get("calendar") or kv.get("where") or "Home"
	try:
		mins = int(float(kv.get("duration") or 60))
	except ValueError:
		mins = 60
	if mins <= 0:
		mins = 60
	ev = Yuna_Calendar().add_event(title, start, start + datetime.timedelta(minutes=mins), location=kv.get("location") or "", calendar=calendar_name)
	return f"calendar: set \"{ev['title']}\" {start.strftime('%Y-%m-%d %H:%M')} ({mins}m) on {ev['calendar']}"


def _action_calendar_get(kv):
	start = _resolve_local(kv.get("date") or "today", "00:00")
	events = Yuna_Calendar().get_events(start, start + datetime.timedelta(days=1))
	if not events:
		return f"calendar: nothing on {start.date().isoformat()}"
	lines = [f"calendar {start.date().isoformat()}:"]
	for e in events[:12]:
		lines.append(f"- {e['start']} {e['title']} ({e['calendar']})")
	return "\n".join(lines)


def _weatherkit_line(loc, date):
	g = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={quote(loc)}&count=1&language=en&format=json", timeout=10, verify=YUNA_TLS_VERIFY).json()
	results = g.get("results") or []
	if not results:
		return None
	lat, lon = results[0]["latitude"], results[0]["longitude"]
	name = results[0].get("name") or loc
	raw = Yuna_Weather().get_weather_at(lat, lon)
	if not isinstance(raw, dict) or raw.get("error") or "currentWeather" not in raw:
		return None
	cw = raw.get("currentWeather") or {}
	now_line = f"{name}: {round(cw.get('temperature', 0), 1)}°C, {cw.get('conditionCode')}, wind {round(cw.get('windSpeed', 0), 1)} km/h"
	day = _resolve_day(date, ZoneInfo("America/Edmonton"))
	if (date or "today").strip().lower() in ("today", "now", ""):
		return f"weather: {now_line}"
	tz = ZoneInfo("America/Edmonton")
	for d in (raw.get("forecastDaily") or {}).get("days") or []:
		start = d.get("forecastStart") or ""
		if not start:
			continue
		local_day = datetime.datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(tz).date()
		if local_day == day:
			return f"weather: {name} {day.isoformat()}: {d.get('conditionCode')}, high {round(d.get('temperatureMax', 0), 1)}°C / low {round(d.get('temperatureMin', 0), 1)}°C, precip {round(float(d.get('precipitationChance') or 0) * 100)}% (now {round(cw.get('temperature', 0), 1)}°C)"
	return f"weather: {now_line} (no daily match for {day.isoformat()})"


def _action_weather(kv):
	loc = kv.get("location") or "Calgary"
	date = (kv.get("date") or "today").strip().lower()
	try:
		kit = _weatherkit_line(loc, date)
		if kit:
			return kit
	except Exception as e:
		print(f"[GreatSage] weatherkit: {e}")
	try:  # Open-Meteo only if WeatherKit has no key or refuses the call.
		g = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={quote(loc)}&count=1&language=en&format=json", timeout=10, verify=YUNA_TLS_VERIFY).json()
		results = g.get("results") or []
		if not results:
			return f"weather: no geocode for {loc}"
		lat, lon = results[0]["latitude"], results[0]["longitude"]
		name = results[0].get("name") or loc
		w = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=7", timeout=10, verify=YUNA_TLS_VERIFY).json()
		cur = w.get("current") or {}
		codes = {0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 61: "rain", 71: "snow", 95: "thunderstorm"}
		code = cur.get("weather_code")
		cond = codes.get(code, f"code {code}")
		now_line = f"{name}: {cur.get('temperature_2m')}°C, {cond}, wind {cur.get('wind_speed_10m')} km/h"
		if date in ("today", "", "now"):
			return f"weather: {now_line}"
		target = date  # Explicit calendar date — try forecast first, then archive (past days).
		daily = w.get("daily") or {}
		days = daily.get("time") or []
		tmax = daily.get("temperature_2m_max") or []
		tmin = daily.get("temperature_2m_min") or []
		dcodes = daily.get("weather_code") or []
		if date == "tomorrow" and days:
			target = days[1] if len(days) > 1 else days[0]
		for i, d in enumerate(days):
			if d == target or d.startswith(target):
				c = codes.get(dcodes[i] if i < len(dcodes) else None, "?")
				hi = tmax[i] if i < len(tmax) else "?"
				lo = tmin[i] if i < len(tmin) else "?"
				return f"weather: {name} {d}: {c}, high {hi}°C / low {lo}°C (now {cur.get('temperature_2m')}°C)"
		if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
			arch = requests.get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto", timeout=10, verify=YUNA_TLS_VERIFY).json()
			ad = arch.get("daily") or {}
			if ad.get("time"):
				c = codes.get((ad.get("weather_code") or [None])[0], "?")
				hi = (ad.get("temperature_2m_max") or ["?"])[0]
				lo = (ad.get("temperature_2m_min") or ["?"])[0]
				return f"weather: {name} {date}: {c}, high {hi}°C / low {lo}°C"
		return f"weather: {now_line} (no daily match for {date})"
	except Exception as e:
		try:  # Fall through to WeatherKit if configured
			raw = Yuna_Weather().get_weather(loc)
			if isinstance(raw, dict) and raw.get("error"):
				return f"weather: {raw['error']} ({e})"
			cw = (raw or {}).get("currentWeather") or {}
			return f"weather: {loc}: {cw.get('temperature')}°C {cw.get('conditionCode')}"
		except Exception as e2:
			return f"weather: failed — {e}; kit — {e2}"


def _action_currency(kv):
	try:
		amt = float(kv.get("amount") or 0)
	except ValueError:
		return "currency: bad amount"
	frm = (kv.get("from") or "USD").upper()
	to = (kv.get("to") or "JPY").upper()
	if amt <= 0:
		return "currency: amount must be > 0"
	r = requests.get(f"https://api.frankfurter.app/latest?amount={amt}&from={frm}&to={to}", timeout=10, verify=YUNA_TLS_VERIFY)
	if r.status_code != 200:
		return f"currency: HTTP {r.status_code}"
	j = r.json()
	rate = (j.get("rates") or {}).get(to)
	if rate is None:
		return f"currency: no rate {frm}->{to}"
	rounded = round(float(rate), 2 if to != "JPY" else 0)
	return f"currency: {amt} {frm} ≈ {rounded} {to}"


def _load_yuna_list(bag_key, list_key):
	block = bag_get(yuna_json_path(), bag_key, {}) or {}
	if not isinstance(block, dict):
		return []
	return list(block.get(list_key) or [])


def _save_yuna_list(bag_key, list_key, items):
	bag_set(yuna_json_path(), bag_key, {list_key: items})


def _action_timer(kv):
	try:
		mins = float(kv.get("duration") or 0)
	except ValueError:
		return "timer: bad duration"
	label = kv.get("label") or "timer"
	if mins <= 0:
		return "timer: duration must be > 0"
	ends = time.time() + mins * 60
	items = _load_yuna_list(_SAGE_TIMER_KEY, "timers")
	items.append({"label": label, "duration_min": mins, "ends_at": ends, "set_at": time.time()})
	_save_yuna_list(_SAGE_TIMER_KEY, "timers", items[-50:])
	end_local = datetime.datetime.fromtimestamp(ends).strftime("%H:%M")
	return f"timer: set {int(mins) if mins == int(mins) else mins}m \"{label}\" (rings ~{end_local})"


def _action_reminder(kv):
	event = kv.get("event") or "reminder"
	date = kv.get("date") or ""
	t = kv.get("time") or ""
	items = _load_yuna_list(_SAGE_REMINDER_KEY, "reminders")
	items.append({"kind": "reminder", "event": event, "date": date, "time": t, "set_at": time.time()})
	_save_yuna_list(_SAGE_REMINDER_KEY, "reminders", items[-100:])
	return f"reminder: '{event}' @ {date} {t}".strip()


def _action_alarm(kv):
	t = kv.get("time") or ""
	label = kv.get("label") or "alarm"
	items = _load_yuna_list(_SAGE_REMINDER_KEY, "reminders")
	items.append({"kind": "alarm", "time": t, "label": label, "set_at": time.time()})
	_save_yuna_list(_SAGE_REMINDER_KEY, "reminders", items[-100:])
	return f"alarm: {t} \"{label}\""


def _action_math(expr):
	expr = (expr or "").strip()
	if not expr:
		return "math: empty expression"
	try:
		tree = ast.parse(expr, mode="eval")
	except SyntaxError as e:
		return f"math: bad expression ({e})"

	def ev(n):
		if isinstance(n, ast.Expression):
			return ev(n.body)
		if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
			return n.value
		if isinstance(n, ast.UnaryOp) and type(n.op) in _MATH_OPS:
			return _MATH_OPS[type(n.op)](ev(n.operand))
		if isinstance(n, ast.BinOp) and type(n.op) in _MATH_OPS:
			return _MATH_OPS[type(n.op)](ev(n.left), ev(n.right))
		raise ValueError("unsupported")

	try:
		val = ev(tree)
	except Exception as e:
		return f"math: {e}"
	if isinstance(val, float) and val.is_integer():
		val = int(val)
	return f"math: {expr} = {val}"


def _action_translate(kv):
	text = kv.get("text") or ""
	lang = (kv.get("targetlanguage") or kv.get("target_language") or "en").strip()
	if not text:
		return "translate: empty text"
	langpair = f"autodetect|{lang}" if len(lang) <= 5 else f"en|{lang[:2].lower()}"  # MyMemory free endpoint — no key required for light use.
	if lang.lower() in ("japanese", "ja", "jp"):
		langpair = "en|ja"
	elif lang.lower() in ("english", "en"):
		langpair = "ja|en"
	try:
		r = requests.get(f"https://api.mymemory.translated.net/get?q={quote(text)}&langpair={quote(langpair)}", timeout=12, verify=YUNA_TLS_VERIFY)
		j = r.json()
		out = ((j.get("responseData") or {}).get("translatedText") or "").strip()
		return f"translate ({lang}): {out}" if out else "translate: empty result"
	except Exception as e:
		return f"translate: {e}"


def _action_dictionary(kv):
	q = kv.get("query") or kv.get("word") or kv.get("topic") or _strip_quotes(kv.get("q") or "")
	if not q:
		raw = " ".join(kv.values()) if kv else ""
		q = _strip_quotes(raw)
	if not q:
		return "dictionary: empty word"
	lang = (kv.get("lang") or kv.get("language") or "").strip()
	hit = dictionary_lookup(q, lang=lang, limit=6)
	lines = hit.get("lines") or []
	if not lines:
		return f"dictionary: no entry for {q}"
	head = hit.get("key") or q
	lg = hit.get("lang") or ""
	return f"dictionary {lg} {head}:\n" + "\n".join(lines[:6])


def _action_summarize(kv):
	url = (kv.get("url") or "").strip()
	text = (kv.get("text") or "").strip()
	if url:
		title, content = WebParser().parse(url=url, output="markdown")
		body = (content or "").strip()
		if len(body) > 3500:
			body = body[:3500].rstrip() + "…"
		if not body:
			return f"summarize: empty page {url}"
		return f"page: {title or url}\nURL: {url}\n\n{body}"
	if text:
		return "text:\n" + (text[:3500] + ("…" if len(text) > 3500 else ""))
	return "summarize: need url or text"


def _action_time(kv):
	name = (kv.get("timezone") or kv.get("tz") or "America/Edmonton").strip() or "America/Edmonton"
	tz = ZoneInfo(name)
	now = datetime.datetime.now(tz)
	return f"time: {now.strftime('%A %Y-%m-%d %H:%M')} {name}"


def _action_news(kv):
	q = kv.get("query") or kv.get("topic") or kv.get("q") or ""
	cat = kv.get("category") or kv.get("source") or None
	query = q or cat or "Calgary"
	try:
		r = requests.get(f"{_search_base_url()}/api/search/news", params={"q": query}, timeout=15, verify=YUNA_TLS_VERIFY, headers={"User-Agent": "Yuna/GreatSage"})
		if r.status_code == 200:
			rows = (r.json() or {}).get("data") or []
			lines = []
			for i in rows[:8]:
				if not isinstance(i, dict):
					continue
				title = i.get("title") or "?"
				src = i.get("source") or ""
				lines.append("- " + title + (f" ({src})" if src else ""))
			if lines:
				return "news:\n" + "\n".join(lines)
	except Exception as e:
		print(f"[GreatSage] news engine: {e}")
	try:
		news = Yuna_News()
		items = news.get_feed(q=q, cat=cat, yuna_only=False)[:8]
		if not items and cat:
			items = news.get_world_news(cat)[:8]
		if not items:
			items = news.get_world_news(q or cat or "technology")[:8]
		if not items:
			return "news: no headlines"
		lines = []
		for i in items:
			title = i.get("title") or "?"
			src = i.get("source") or i.get("category") or ""
			lines.append(f"- {title}" + (f" ({src})" if src else ""))
		return "news:\n" + "\n".join(lines)
	except Exception as e:
		return f"news: {e}"


def _action_send_text(kv):
	to = kv.get("recipient") or "?"  # Soft ack only — no silent SMS without an explicit provider hook.
	msg = kv.get("message") or ""
	return f"sendText: queued to {to} — \"{msg}\" (delivery provider not linked; treat as draft)"


def _action_pizza(kv):
	typ = kv.get("type") or "Margherita"
	size = kv.get("size") or "medium"
	extras = kv.get("extras") or "none"
	return f"pizza: {size} {typ} (+{extras}) — order noted (no live storefront linked; confirm before paying)"


def _search_base_url():
	"""Base URL for Yuna Search. Override with ``YUNA_SEARCH_URL``. A retired host is repointed at the live one and announced once, because leaving it alone means every Great Sage search dies on a TLS alert instead of a clear error."""
	base = os.environ.get("YUNA_SEARCH_URL", SEARCH_URL_DEFAULT).rstrip("/")
	if urlparse(base).netloc.lower() in SEARCH_HOSTS_RETIRED:
		if base not in _SEARCH_URL_WARNED:
			_SEARCH_URL_WARNED.add(base)
			print(f"[YunaSearch] YUNA_SEARCH_URL={base!r} is retired and no longer serves TLS — using {SEARCH_URL_DEFAULT}")
		return SEARCH_URL_DEFAULT
	return base


def _search_host():
	"""Host of Yuna Search, derived so the self-scrape guard cannot go stale."""
	return urlparse(_search_base_url()).netloc.lower()


def _kagi_web_results(data) -> list[dict]:
	"""Pull standard web hits out of a Kagi / Yuna Search JSON payload. Supports legacy v0 list payloads and v1 ``data.search`` buckets."""
	if not isinstance(data, dict):
		return []
	raw = data.get("data")
	if isinstance(raw, dict):
		items = raw.get("search") or []
	elif isinstance(raw, list):
		items = raw
	else:
		return []
	out = []
	for item in items:
		if not isinstance(item, dict):
			continue
		t = item.get("t")
		if t not in (None, 0, "0"):
			continue
		url = item.get("url")
		if isinstance(url, str) and url.startswith("http"):
			out.append(item)
	return out


def _cap_text(text: str, char_limit: int) -> str:
	text = (text or "").strip()
	if len(text) > char_limit:
		return text[:char_limit].rstrip() + "…"
	return text


def _format_snippet(title: str, snippet: str) -> str:
	title = (title or "").strip()
	snippet = (snippet or "").strip()
	if title and snippet:
		return f"{title}\n\n{snippet}"
	return title or snippet


def _is_usable_content(text: str) -> bool:
	"""Reject reference lists / citation footers that WebParser sometimes grabs."""
	if not text or len(text.strip()) < 80:
		return False
	if text.count("cite_ref") >= 2:
		return False
	if re.search(r"^\s*[-•*]\s*\^", text, re.MULTILINE):
		return False
	if re.search(r"\[\*\*\*[a-z]\*\*\*\]", text, re.IGNORECASE):
		return False
	links = len(re.findall(r"\[[^\]]+\]\(https?://", text))
	if links >= 4 and links * 40 > len(text):
		return False
	return True


def _parse_outbound_page(url: str, fallback_title: str = "") -> str:
	"""Fetch and parse one outbound result URL — never the Yuna Search host."""
	host = _search_host()
	if not url or (host and host in url.lower()):
		return ""
	try:
		page_title, content = WebParser().parse(url=url, timeout=15)
	except Exception as e:
		print(f"[GreatSage.search] WebParser error on {url}: {e}")
		traceback.print_exc()
		return ""

	text = (content or "").strip()
	heading = (page_title or fallback_title or "").strip()
	if heading and heading not in text[:200]:
		text = f"{heading}\n\n{text}".strip()
	return text if _is_usable_content(text) else ""


def _action_search(query: str, char_limit: int = 1000) -> str:
	"""Yuna Search ``/api/search`` JSON → first good outbound URL → WebParser. Never scrapes the Yuna Search HTML shell. If page parsing fails or looks like citation junk, tries the next API hit, then falls back to API snippets."""
	query = (query or "").strip()
	if not query:
		print("[GreatSage.search] empty query, bailing")
		return "(empty search query)"

	api_url = f"{_search_base_url()}/api/search?q={quote(query)}"
	print(f"[GreatSage.search] GET {api_url}")
	try:
		resp = requests.get(api_url, timeout=15, verify=YUNA_TLS_VERIFY, headers={"User-Agent": "Mozilla/5.0 (compatible; Yuna/GreatSage)"})
		print(f"[GreatSage.search] status={resp.status_code} ctype={resp.headers.get('content-type','')!r}")
		if resp.status_code != 200:
			return f"(search failed: HTTP {resp.status_code})"
	except Exception as e:
		print(f"[GreatSage.search] request error: {e}")
		traceback.print_exc()
		return f"(search error: {e})"

	try:
		payload = resp.json()
	except Exception as e:
		print(f"[GreatSage.search] JSON decode failed: {e}")
		return "(search returned non-JSON response)"

	if isinstance(payload, dict) and payload.get("error"):
		err = payload.get("error")
		print(f"[GreatSage.search] API error: {err!r}")
		return f"(search error: {err})"

	results = _kagi_web_results(payload)
	print(f"[GreatSage.search] web hits={len(results)}")
	if not results:
		return "(no search results)"

	best_snippet = ""
	own_host = _search_host()
	for idx, item in enumerate(results[:5]):
		title = (item.get("title") or "").strip()
		snippet = (item.get("snippet") or "").strip()
		formatted = _format_snippet(title, snippet)
		if len(formatted) > len(best_snippet):
			best_snippet = formatted

		result_url = (item.get("url") or "").strip()
		if not result_url or (own_host and own_host in result_url.lower()):
			continue

		print(f"[GreatSage.search] try #{idx + 1} url={result_url!r}")
		parsed = _parse_outbound_page(result_url, fallback_title=title)
		if parsed:
			out = _cap_text(parsed, char_limit)
			print(f"[GreatSage.search] returning parsed page ({len(out)} chars) from {result_url!r}")
			return out

	if best_snippet:
		out = _cap_text(best_snippet, char_limit)
		print(f"[GreatSage.search] returning API snippet ({len(out)} chars)")
		return out

	print("[GreatSage.search] no usable page or snippet")
	return "(could not extract content from search results)"
