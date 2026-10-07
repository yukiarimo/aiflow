import json
import os
import re
import threading
import time

_LOCK = threading.Lock()
_ENTRIES = None
POS_SHORT = {"noun": "n.", "verb": "v.", "adjective": "adj.", "adverb": "adv.", "transitive verb": "v.", "intransitive verb": "v.", "preposition": "prep.", "conjunction": "conj.", "interjection": "interj.", "pronoun": "pron.", "ж.": "ж.", "м.": "м.", "ср.": "ср.", "名詞": "名", "形容詞": "形", "副詞": "副", "動詞": "動"}


def dictionary_path():
	return os.environ.get("YUNA_DICTIONARY") or os.path.join("lib", "yuna_dictionary.json")


def _norm(s, lang):
	s = (s or "").strip()
	if lang in ("en", "ru"):
		return s.casefold()
	return s


def _detect(q):
	if re.search(r"[\u0400-\u04FF]", q):
		return "ru"
	if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf]", q):
		return "ja"
	return "en"


def _fmt_pos(pos):
	if not pos:
		return ""
	p = pos.strip().lower()
	return POS_SHORT.get(p, POS_SHORT.get(pos, pos))


def _line(sense):
	text = (sense.get("text") or "").strip()
	if not text:
		return ""
	kind = sense.get("kind") or "definition"
	pos = _fmt_pos(sense.get("pos") or "")
	lt = (sense.get("lang_to") or "").upper()
	if kind == "translation" and lt:
		return f"→ {lt}{f' ({pos})' if pos else ''}: {text}"
	if kind == "synonym":
		return f"≈ {text}"
	if kind == "example":
		return f"▸ {text}"
	if kind == "glossary":
		return text
	if pos:
		return f"({pos}) {text}"
	return text


def ensure():
	"""Load the book once. Later calls return the same dict."""
	global _ENTRIES
	with _LOCK:
		if _ENTRIES is not None:
			return _ENTRIES
		path = dictionary_path()
		t0 = time.perf_counter()
		if not os.path.isfile(path):
			_ENTRIES = {}
			print(f"dictionary: missing {path}", flush=True)
			return _ENTRIES
		with open(path, encoding="utf-8") as f:
			data = json.load(f)
		entries = data.get("entries") if isinstance(data, dict) else None
		_ENTRIES = entries if isinstance(entries, dict) else {}
		print(f"dictionary: {len(_ENTRIES)} entries in {time.perf_counter() - t0:.2f}s", flush=True)
		return _ENTRIES


def warm():
	threading.Thread(target=ensure, name="yuna-dictionary", daemon=True).start()


def lookup(q, lang="", limit=14):
	q = (q or "").strip()
	if not q:
		return {"q": "", "lang": "", "key": "", "senses": [], "lines": [], "more": 0}
	entries = ensure()
	lang = (lang or "").strip().lower()
	order = [lang] if lang in ("en", "ru", "ja") else [_detect(q), "en", "ru", "ja"]
	seen = set()
	langs = []
	for lg in order:
		if lg in seen or lg not in ("en", "ru", "ja"):
			continue
		seen.add(lg)
		langs.append(lg)
	if lang in ("en", "ru", "ja"):
		langs = [lang]
	for lg in langs:
		hit = entries.get(f"{lg}:{_norm(q, lg)}")
		if not hit:
			continue
		senses = [s for s in (hit.get("senses") or []) if isinstance(s, dict) and (s.get("text") or "").strip()]
		lines = [ln for ln in (_line(s) for s in senses[:limit]) if ln]
		return {"q": q, "lang": hit.get("lang") or lg, "key": hit.get("key") or q, "senses": senses[:limit], "lines": lines, "more": max(0, len(senses) - limit)}
	return {"q": q, "lang": langs[0] if langs else "", "key": "", "senses": [], "lines": [], "more": 0}
