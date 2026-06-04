"""LoliConnect action dispatcher.

When Himitsu (or any other loli) emits a block like::

    <action>LoliConnect(
        search -> "Elon Musk early life"
    )</action>

we parse the function name + args out of that block and dispatch to a Python
implementation. The result is folded back into Himitsu's stream as a
``<data>...</data>`` block so she can keep generating with the new context.

For now only ``search`` is implemented (Yuna Search → first result → web
parser → 1000-char snippet). The parser/dispatcher is intentionally tolerant
of casing and whitespace because the model often jitters those.
"""

from __future__ import annotations

import os
import re
import traceback
from urllib.parse import quote

import requests

from .utils import WebParser


# --------------------------- Action parsing ---------------------------

# Matches a `<action>LoliConnect( <name> -> <args> )</action>` block.
# - Case-insensitive everywhere (`Search`, `SEARCH`, `LOLICONNECT`, ...).
# - Tolerates any whitespace/tab/newline around tokens.
# - Captures the function name and the raw args string (anything up to the
#   matching close-paren, lazily).
# - `</action>` is optional because Himitsu phase-1 generation stops *on*
#   that token and the VLM trims it from the streamed text.
ACTION_REGEX = re.compile(
	r"<action>\s*LoliConnect\s*\(\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*->\s*(?P<args>.*?)\s*\)\s*(?:</action>)?",
	re.IGNORECASE | re.DOTALL,
)

ACTION_OPEN_RE = re.compile(r"<action>\s*LoliConnect\s*\(", re.IGNORECASE)


def parse_action(text: str):
	"""Return ``(name_lower, args_str, full_match_str)`` for the first
	LoliConnect action block in ``text``, or ``None`` if no action is found.
	"""
	if not text:
		return None
	m = ACTION_REGEX.search(text)
	if not m:
		return None
	return (
		m.group("name").strip().lower(),
		m.group("args").strip(),
		m.group(0),
	)


def close_action_tag(text: str) -> str:
	"""Re-append ``</action>`` when generation stopped on that token."""
	if not text or "</action>" in text:
		return text or ""
	if ACTION_OPEN_RE.search(text) and ACTION_REGEX.search(text):
		return text.rstrip() + "</action>"
	return text


def _strip_quotes(s: str) -> str:
	s = (s or "").strip()
	if len(s) >= 2 and s[0] in "\"'“”‘’" and s[-1] in "\"'“”‘’":
		return s[1:-1].strip()
	return s


# --------------------------- Dispatch ---------------------------

def dispatch_action(name: str, args: str, char_limit: int = 1000) -> str:
	"""Run the named action and return a plain-text snippet (≤ char_limit).
	Always returns a string — never raises — so the calling stream can keep
	going even if the action fails. Prints debug info to stdout."""
	name = (name or "").strip().lower()
	print(f"[LoliConnect] dispatch: name={name!r}  args={args!r}")

	if name == "search":
		query = _strip_quotes(args)
		return _action_search(query, char_limit=char_limit)

	# Unknown / unimplemented action — degrade gracefully so the model can
	# wrap up its message instead of dying.
	msg = f"(LoliConnect: action {name!r} is not implemented yet)"
	print(f"[LoliConnect] {msg}")
	return msg


# --------------------------- Search ---------------------------

def _search_base_url() -> str:
	"""Base URL for Yuna Search (Flask app in yuna-ai/search.py).

	Override with ``YUNA_SEARCH_URL`` — e.g. ``https://search.yuna-ai.com``
	on Tailscale, or ``http://127.0.0.1:2519`` when running ``--mode all`` locally.
	"""
	return os.environ.get("YUNA_SEARCH_URL", "https://search.yuna-ai.com").rstrip("/")


def _kagi_web_results(data) -> list[dict]:
	"""Pull standard web hits out of a Kagi / Yuna Search JSON payload.

	Supports legacy v0 list payloads and v1 ``data.search`` buckets.
	"""
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
	if not url or "search.yuna-ai.com" in url:
		return ""
	try:
		page_title, content = WebParser().parse(url=url, timeout=15)
	except Exception as e:
		print(f"[LoliConnect.search] WebParser error on {url}: {e}")
		traceback.print_exc()
		return ""

	text = (content or "").strip()
	heading = (page_title or fallback_title or "").strip()
	if heading and heading not in text[:200]:
		text = f"{heading}\n\n{text}".strip()
	return text if _is_usable_content(text) else ""


def _action_search(query: str, char_limit: int = 1000) -> str:
	"""Yuna Search ``/api/search`` JSON → first good outbound URL → WebParser.

	Never scrapes the Yuna Search HTML shell. If page parsing fails or looks
	like citation junk, tries the next API hit, then falls back to API snippets.
	"""
	query = (query or "").strip()
	if not query:
		print("[LoliConnect.search] empty query, bailing")
		return "(empty search query)"

	api_url = f"{_search_base_url()}/api/search?q={quote(query)}"
	print(f"[LoliConnect.search] GET {api_url}")
	try:
		resp = requests.get(
			api_url,
			timeout=15,
			verify=False,
			headers={"User-Agent": "Mozilla/5.0 (compatible; Yuna/Himitsu)"},
		)
		print(f"[LoliConnect.search] status={resp.status_code} ctype={resp.headers.get('content-type','')!r}")
		if resp.status_code != 200:
			return f"(search failed: HTTP {resp.status_code})"
	except Exception as e:
		print(f"[LoliConnect.search] request error: {e}")
		traceback.print_exc()
		return f"(search error: {e})"

	try:
		payload = resp.json()
	except Exception as e:
		print(f"[LoliConnect.search] JSON decode failed: {e}")
		return "(search returned non-JSON response)"

	if isinstance(payload, dict) and payload.get("error"):
		err = payload.get("error")
		print(f"[LoliConnect.search] API error: {err!r}")
		return f"(search error: {err})"

	results = _kagi_web_results(payload)
	print(f"[LoliConnect.search] web hits={len(results)}")
	if not results:
		return "(no search results)"

	best_snippet = ""
	for idx, item in enumerate(results[:5]):
		title = (item.get("title") or "").strip()
		snippet = (item.get("snippet") or "").strip()
		formatted = _format_snippet(title, snippet)
		if len(formatted) > len(best_snippet):
			best_snippet = formatted

		result_url = (item.get("url") or "").strip()
		if not result_url or "search.yuna-ai.com" in result_url:
			continue

		print(f"[LoliConnect.search] try #{idx + 1} url={result_url!r}")
		parsed = _parse_outbound_page(result_url, fallback_title=title)
		if parsed:
			out = _cap_text(parsed, char_limit)
			print(f"[LoliConnect.search] returning parsed page ({len(out)} chars) from {result_url!r}")
			return out

	if best_snippet:
		out = _cap_text(best_snippet, char_limit)
		print(f"[LoliConnect.search] returning API snippet ({len(out)} chars)")
		return out

	print("[LoliConnect.search] no usable page or snippet")
	return "(could not extract content from search results)"
