# drop-in of yuna_speech/text.py — tables in data/*.json, G2P via vendored espeak 1.48 + OpenJTalk
import ctypes, importlib.util, json, os, re, sys, types
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"


def _j(name):
	return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


_SYM = _j("symbols.json")
_CLN = _j("cleaners.json")
_JA = _j("ja_rewrite.json")
_LANG = _j("lang.json")
_LNG = _j("langs.json")
_VOICES = _LNG["voices"]
_FILES = _LNG.get("files", {})
_ALIASES = _LNG["aliases"]
_PUNCT = _j("punct.json")
symbols = list(_SYM["symbols"])
assert len(symbols) == int(_SYM["count"]) == 177
_SYMBOL_TO_ID = {ch: i for i, ch in enumerate(symbols)}
assert list(_SYMBOL_TO_ID.values()) == list(range(len(_SYMBOL_TO_ID)))
SPACE_ID = _SYMBOL_TO_ID[" "]
_symbol_to_id = _SYMBOL_TO_ID
_id_to_symbol = {i: ch for ch, i in _SYMBOL_TO_ID.items()}
_PHONE_FOLD = ((re.compile("-+"), "\u2014"), ("^", "\u02b2"))  # fold g2p dupes onto table symbols: `--`→`—` (ja em-dash = en/ru pause), `^`→`ʲ` (ru soft sign); both were dropped before → 20666 phones the encoder never saw


def fold_phones(cleaned_text):
	"""Rewrite g2p spellings onto their table equivalents. Idempotent."""
	for pat, rep in _PHONE_FOLD:
		cleaned_text = pat.sub(rep, cleaned_text) if hasattr(pat, "sub") else cleaned_text.replace(pat, rep)
	return cleaned_text


_TAGS = _j("tags.json")  # [happy]/[laugh]/… — styles colour speech; events are duration phones (177 unused rows); G2P must not see brackets
EVENT_SYMBOLS = dict(_TAGS["events"])
STYLE_TAGS = set(_TAGS["styles"])
EVENT_TAGS = set(EVENT_SYMBOLS)
EVENT_RESIDUE = {k: (v[0], float(v[1])) for k, v in _TAGS.get("residue", {}).items()}
_TAG_RE = re.compile(r"\[([^\]]+)\]")


def normalize_tag(name):
	return " ".join((name or "").strip().lower().split())


def iter_tagged(text):
	"""Yield ('text'|'event'|'style'|'unknown', payload). Skips whitespace right after `]`."""
	pos = 0
	text = text or ""
	for m in _TAG_RE.finditer(text):
		if m.start() > pos:
			yield ("text", text[pos:m.start()])
		name = normalize_tag(m.group(1))
		if name in EVENT_TAGS:
			yield ("event", name)
		elif name in STYLE_TAGS:
			yield ("style", name)
		else:
			yield ("unknown", name)
		pos = m.end()
		while pos < len(text) and text[pos].isspace():
			pos += 1
	if pos < len(text):
		yield ("text", text[pos:])


def has_tags(text):
	return bool(_TAG_RE.search(text or ""))


def expand_event_tags(text):
	"""Replace `[laugh]` etc. with their IPA-table symbols; leave style tags alone."""
	def repl(m):
		name = normalize_tag(m.group(1))
		return EVENT_SYMBOLS.get(name, m.group(0))

	return _TAG_RE.sub(repl, text or "")


def strip_style_tags(text):
	"""Remove `[happy]` / `[neutral]` … (and unknown brackets). Keep event symbols if already expanded."""
	def repl(m):
		name = normalize_tag(m.group(1))
		if name in EVENT_TAGS:
			return m.group(0)
		return ""

	out = _TAG_RE.sub(repl, text or "")
	return re.sub(r"  +", " ", out).strip()


def materialize_tags_for_ids(text):
	"""IPA/orthography with brackets → phone string for the 177 table (events→symbols, styles gone)."""
	return fold_phones(strip_style_tags(expand_event_tags(text)))


def text_to_phonemes_tagged(text, language="en-us", language_map=None):
	"""G2P keeping mid-line `[tag]`s: `[neutral] Hello [laugh] there.` → `[neutral] həlˈoʊ [laugh] ðˈɛɹ.` — for .cleaned filelists so StyleTTS2 recovers emotion + event phones."""
	if language_map is None:
		language_map = _LANG_MAP
	lang = language_map.get(language, language)
	if not has_tags(text):
		return _g2p((text or "").strip(), lang).strip()
	chunks = []
	for kind, payload in iter_tagged(text):
		if kind == "text":
			body = payload.strip()
			if not body:
				continue
			ph = _g2p(body, lang).strip()
			if ph:
				chunks.append(ph)
		elif kind in ("event", "style"):
			chunks.append("[%s]" % payload)
		else:
			pass  # unknown tags dropped
	return " ".join(chunks)


RUSSIAN_ALPHABET = set(_LANG["russian"])
JAPANESE_HIRAGANA = set(_LANG["hiragana"])
JAPANESE_KATAKANA = set(_LANG["katakana"])
JAPANESE_KANJI_RANGES = [tuple(x) for x in _LANG["kanji_ranges"]]

_abbreviations = [(re.compile(p, re.IGNORECASE), r) for p, r in _CLN["abbreviations"]]
CURRENCY_MAP = _CLN["currency_map"]
SUFFIX_MAP = _CLN["suffix_map"]
_INITIALS_RE = re.compile(_CLN["initials_re"])
_CURRENCY_RE = re.compile(_CLN["currency_re"])
_CURRENCY_REV_RE = re.compile(_CLN["currency_rev_re"])
_SUFFIX_NUM_RE = re.compile(_CLN["suffix_num_re"])
_SCI_RE = re.compile(_CLN["sci_re"])
_RANGE_RE = re.compile(_CLN["range_re"])
_DECIMAL_RE = re.compile(_CLN["decimal_re"])
_URLS_RE = re.compile(_CLN["urls_re"])
_NUM_COMMAS_RE = re.compile(_CLN["number_commas_re"])
_ROMAN_VALUES = {k: int(v) for k, v in _CLN["roman_values"].items()}
_ROMAN_EXCLUDE = set(_CLN["roman_exclude"])
_ROMAN_VALID = re.compile(_CLN["roman_valid_re"])
_ROMAN_MATCH = re.compile(_CLN["roman_match_re"])
_LANG_MAP = {**_CLN["language_map"], **_ALIASES}
_PHONE_SWAP = dict(_JA["phone_replace"])

_japanese_characters = re.compile(_JA["japanese_characters"])
_japanese_marks = re.compile(_JA["japanese_marks"])
_symbols_to_japanese = [(re.compile(p), r) for p, r in _JA["symbols_to_japanese"]]
_romaji_to_ipa = [(re.compile(p), r) for p, r in _JA["romaji_to_ipa"]]
_real_sokuon = [(re.compile(p), r) for p, r in _JA["real_sokuon"]]
_real_hatsuon = [(re.compile(p), r) for p, r in _JA["real_hatsuon"]]
_PHONEME_RE = re.compile(_JA["phoneme_re"])
_A1_RE = re.compile(_JA["a1_re"])
_A2_RE = re.compile(_JA["a2_re"])
_A3_RE = re.compile(_JA["a3_re"])
_VOWEL_GEM = re.compile(_JA["vowel_geminate_re"])
_ASPIRATE = re.compile(_JA["aspirate_re"])
_SKIP = set(_JA["skip_phones"])
_DICT_NAME = _JA["openjtalk_dict"]
_UNICODE = None
_ESPEAK = None
_JTALK = None
_MARKS = _PUNCT["default_marks"]
_FLAG_RE = re.compile(_PUNCT["language_switch_re"])
_POST_US = re.compile(r"_+")
_POST_USP = re.compile(r"_ ")
_TEXT_MODE = int(_PUNCT["text_mode"])
_PHONEMES_MODE = int(_PUNCT["phonemes_mode_1_48"])
_AUDIO_SYNC = int(_PUNCT["audio_output_synchronous"])
MarkIndex = namedtuple("MarkIndex", "index mark position")


def unidecode(text):
	global _UNICODE
	if _UNICODE is None:
		raw = _j("unicode.json")
		_UNICODE = {k: v for k, v in raw.items() if not str(k).startswith("_")}
		if not _UNICODE: raise RuntimeError("data/unicode.json missing — python trace.py --unicode")
	out = []
	for ch in text:
		key = f"{ord(ch):04X}"
		if key in _UNICODE: out.append(_UNICODE[key])
		elif ord(ch) < 128: out.append(ch)
		else: out.append("")
	return "".join(out)


class Punct:
	def __init__(self, marks=_MARKS):
		self.marks = "".join(set(marks))
		self.marks_re = re.compile(fr'(\s*[{re.escape(self.marks)}]+\s*)+')

	def preserve(self, line):
		matches = list(re.finditer(self.marks_re, line))
		if not matches: return [line], []
		if len(matches) == 1 and matches[0].group() == line: return [], [MarkIndex(0, line, "A")]
		marks = []
		for match in matches:
			position = "I"
			if match == matches[0] and line.startswith(match.group()): position = "B"
			elif match == matches[-1] and line.endswith(match.group()): position = "E"
			marks.append(MarkIndex(0, match.group(), position))
		chunks = []
		for mark in marks:
			split = line.split(mark.mark)
			prefix, suffix = split[0], mark.mark.join(split[1:])
			chunks.append(prefix)
			line = suffix
		return [c for c in chunks + [line] if c], marks

	def restore(self, text, marks, strip=False):
		text = list(text)
		out, pos = [], 0
		while text or marks:
			if not marks:
				for line in text:
					if not strip and not line.endswith(" "): line = line + " "
					out.append(line)
				text = []
			elif not text:
				out.append("".join(m.mark for m in marks))
				marks = []
			else:
				cur = marks[0]
				if cur.index == pos:
					mark = marks[0].mark
					marks = marks[1:]
					if text[0].endswith(" "): text[0] = text[0][:-1]
					if cur.position == "B": text[0] = mark + text[0]
					elif cur.position == "E":
						out.append(text[0] + mark + ("" if strip or mark.endswith(" ") else " "))
						text = text[1:]
						pos += 1
					elif cur.position == "A":
						out.append(mark + ("" if strip or mark.endswith(" ") else " "))
						pos += 1
					else:
						if len(text) == 1: text[0] = text[0] + mark
						else:
							first, text = text[0], text[1:]
							text[0] = first + mark + text[0]
				else:
					out.append(text[0])
					text = text[1:]
					pos += 1
		return out


class EspeakIPA:
	def __init__(self):
		dylib = LIB / "libespeak.dylib"
		if not dylib.is_file(): raise RuntimeError("lib/libespeak.dylib missing — python trace.py --lib")
		if not (LIB / "espeak-data" / "phontab").exists(): raise RuntimeError("lib/espeak-data missing — python trace.py --lib")
		self._lib = ctypes.CDLL(str(dylib))
		if self._lib.espeak_Initialize(_AUDIO_SYNC, 0, str(LIB).encode(), 0) <= 0: raise RuntimeError("espeak_Initialize failed")
		self._lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
		self._lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
		self._lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
		self._voice = None
		self._punct = Punct()

	def set_voice(self, code):
		if self._voice == code: return
		name = _VOICES.get(code)
		if not name: raise RuntimeError(f"no eSpeak voice for {code}")
		for cand in (name, _FILES.get(code, ""), code):
			if cand and self._lib.espeak_SetVoiceByName(cand.encode()) == 0:
				self._voice = code
				return
		raise RuntimeError(f"espeak_SetVoiceByName({name}|{_FILES.get(code)}|{code}) failed")

	def _raw(self, text):
		text_ptr = ctypes.pointer(ctypes.c_char_p(text.encode("utf8")))
		parts = []
		while text_ptr.contents.value is not None:
			ph = self._lib.espeak_TextToPhonemes(text_ptr, _TEXT_MODE, _PHONEMES_MODE)
			if ph: parts.append(ph.decode())
		return " ".join(parts)

	def _post(self, line):
		line = line.strip().replace("\n", " ").replace("  ", " ")
		line = _POST_US.sub("_", line)
		line = _POST_USP.sub(" ", line)
		if "(" in line and _FLAG_RE.search(line): line = _FLAG_RE.sub("", line)
		if not line: return ""
		return "".join(w.strip().replace("_", "") + " " for w in line.split(" "))

	def phonemize(self, text, language):
		self.set_voice(language)
		if not self._punct.marks_re.search(text): return self._post(self._raw(text))
		chunks, marks = self._punct.preserve(text)
		ph = [self._post(self._raw(c)) for c in chunks]
		restored = self._punct.restore(ph, marks, strip=False)
		return "".join(restored) if restored else ""


def espeak_ipa(text, language):
	global _ESPEAK
	if _ESPEAK is None: _ESPEAK = EspeakIPA()
	return _ESPEAK.phonemize(text, language)


def _site():
	return Path(os.environ.get("YUNA_PYOPENJTALK", "/opt/homebrew/Caskroom/miniforge/base/envs/yuna/lib/python3.12/site-packages/pyopenjtalk"))


def _so():
	hits = list(_site().glob("openjtalk*.so"))
	if hits: return hits[0]
	raise RuntimeError("openjtalk*.so not found — site-packages pyopenjtalk (do not copy the .so)")


def _dict_dir():
	p = LIB / "ja" / _DICT_NAME
	if p.is_symlink(): raise RuntimeError("lib/ja dict is a symlink — python trace.py --flatten")
	if (p / "sys.dic").is_file(): return str(p)
	raise RuntimeError("OpenJTalk dict missing — python trace.py --lib")


def _load_mod():
	if "pyopenjtalk.openjtalk" in sys.modules: return sys.modules["pyopenjtalk.openjtalk"]
	so = _so()
	pkg = sys.modules.get("pyopenjtalk")
	if pkg is not None and getattr(pkg, "__file__", None) and Path(pkg.__file__).name == "__init__.py":
		from pyopenjtalk import openjtalk as mod
		return mod
	if pkg is None:
		pkg = types.ModuleType("pyopenjtalk")
		pkg.__path__ = [str(so.parent)]
		sys.modules["pyopenjtalk"] = pkg
	spec = importlib.util.spec_from_file_location("pyopenjtalk.openjtalk", so)
	mod = importlib.util.module_from_spec(spec)
	sys.modules["pyopenjtalk.openjtalk"] = mod
	spec.loader.exec_module(mod)
	return mod


def extract_fullcontext(text):
	global _JTALK
	if _JTALK is None:
		mod = _load_mod()
		dn = _dict_dir()
		jtalk = mod.OpenJTalk(dn_mecab=dn.encode("utf-8") if isinstance(dn, str) else dn)
		_JTALK = (mod, jtalk, jtalk.make_label)
	mod, jtalk, make_label = _JTALK
	return make_label(jtalk.run_frontend(text))


def load_filepaths_and_text(filename, split="|"):
	with open(filename, encoding="utf-8") as f:
		return [line.strip().split(split) for line in f]


def symbols_to_japanese(text):
	for regex, replacement in _symbols_to_japanese:
		text = re.sub(regex, replacement, text)
	return text


def japanese_to_romaji_with_accent(text):
	text = symbols_to_japanese(text)
	sentences = re.split(_japanese_marks, text)
	marks = re.findall(_japanese_marks, text)
	out_text = ""
	for i, sentence in enumerate(sentences):
		if re.match(_japanese_characters, sentence):
			if out_text != "": out_text += " "
			labels = extract_fullcontext(sentence)
			for n, label in enumerate(labels):
				i0 = label.find("-")
				i1 = label.find("+", i0 + 1)
				phoneme = label[i0 + 1:i1]
				if phoneme in _SKIP: continue
				phoneme = _PHONE_SWAP.get(phoneme, phoneme)
				out_text += phoneme
				a1 = int(_A1_RE.search(label).group(1))
				a2 = int(_A2_RE.search(label).group(1))
				a3 = int(_A3_RE.search(label).group(1))
				nxt_l = labels[n + 1]
				n0 = nxt_l.find("-")
				nxt = nxt_l[n0 + 1:nxt_l.find("+", n0 + 1)]
				a2_next = -1 if nxt in _SKIP else int(_A2_RE.search(labels[n + 1]).group(1))
				if a3 == 1 and a2_next == 1: out_text += " "
				elif a1 == 0 and a2_next == a2 + 1: out_text += "↓"
				elif a2 == 1 and a2_next == 2: out_text += "↑"
		if i < len(marks):
			out_text += unidecode(marks[i]).replace(" ", "")
	return out_text


def get_real_sokuon(text):
	for regex, replacement in _real_sokuon:
		text = re.sub(regex, replacement, text)
	return text


def get_real_hatsuon(text):
	for regex, replacement in _real_hatsuon:
		text = re.sub(regex, replacement, text)
	return text


def japanese_to_ipa2(text):
	text = japanese_to_romaji_with_accent(text).replace(_JA["ellipsis"][0], _JA["ellipsis"][1])
	text = get_real_sokuon(text)
	text = get_real_hatsuon(text)
	for regex, replacement in _romaji_to_ipa:
		text = re.sub(regex, replacement, text)
	return text


def japanese_to_ipa3(text):
	text = japanese_to_ipa2(text)
	for a, b in _JA["ipa3"].items():
		text = text.replace(a, b)
	text = _VOWEL_GEM.sub(lambda x: x.group(0)[0] + "ː" * (len(x.group(0)) - 1), text)
	text = _ASPIRATE.sub(r"\1ʰ", text)
	return text


def is_japanese(char):
	if char in JAPANESE_HIRAGANA or char in JAPANESE_KATAKANA: return True
	o = ord(char)
	for start, end in JAPANESE_KANJI_RANGES:
		if start <= o <= end: return True
	return False


def is_russian(char):
	return char in RUSSIAN_ALPHABET


def detect_language_for_word(word):
	russian_count = sum(1 for c in word if is_russian(c))
	japanese_count = sum(1 for c in word if is_japanese(c))
	if russian_count > 0 and russian_count >= japanese_count: return "ru"
	elif japanese_count > 0: return "ja"
	return "en-us"


def clean_urls(text):
	return _URLS_RE.sub("", text)


def clean_punctuation(text):
	text = re.sub(r"\s*\.\.\.\s*", "—", text)
	text = re.sub(r"(?:\s*\.\s*){3,}", "—", text)
	text = re.sub(r"\?+!", "?", text)
	text = re.sub(r"!\+?\?", "?", text)
	text = re.sub(r"\?+", "?", text)
	text = re.sub(r"!+", "!", text)
	text = re.sub(r"\s*–\s*", "-", text)
	text = re.sub(r"\s+-\s+", "—", text)
	text = re.sub(r"\s*—\s*", "—", text)
	text = re.sub(r"\*", "", text)
	return text


def replace_roman_numerals(text):
	def roman_to_int(s):
		total = prev = 0
		for char in reversed(s.upper()):
			curr = _ROMAN_VALUES[char]
			total += curr if curr >= prev else -curr
			prev = curr
		return total

	def repl(m):
		word = m.group(0)
		if len(word) == 1: return word
		if word.upper() in _ROMAN_EXCLUDE: return word
		if _ROMAN_VALID.match(word): return str(roman_to_int(word))
		return word

	return _ROMAN_MATCH.sub(repl, text)


def remove_number_commas(text):
	return _NUM_COMMAS_RE.sub("", text)


def process_currencies(text, lang):
	def repl(m, is_rev=False):
		if not is_rev: sym, num_str, suf = m.groups()
		else: num_str, suf, sym = m.groups()
		val = float(num_str)
		c_info = CURRENCY_MAP[sym].get(lang, CURRENCY_MAP[sym]["en-us"])
		is_singular = (val == 1.0 and not suf)
		if lang == "en-us": word = c_info[0] if is_singular else c_info[1]
		elif lang == "ru":
			if suf: word = c_info[2]
			else:
				v, v1 = int(val) % 100, int(val) % 10
				if 11 <= v <= 19: word = c_info[2]
				elif v1 == 1: word = c_info[0]
				elif 2 <= v1 <= 4: word = c_info[1]
				else: word = c_info[2]
		else: word = c_info
		suf_word = ""
		if suf: suf_word = " " + SUFFIX_MAP[suf.upper()].get(lang, SUFFIX_MAP[suf.upper()]["en-us"])
		return f"{num_str}{suf_word} {word}"

	text = _CURRENCY_RE.sub(lambda m: repl(m, False), text)
	text = _CURRENCY_REV_RE.sub(lambda m: repl(m, True), text)
	return text


def expand_numbers_and_symbols(text, lang):
	def repl_suf(m):
		return f"{m.group(1)} {SUFFIX_MAP[m.group(2).upper()].get(lang, SUFFIX_MAP[m.group(2).upper()]['en-us'])}"

	text = _SUFFIX_NUM_RE.sub(repl_suf, text)
	sci_words = _CLN["sci_words"].get(lang, _CLN["sci_words"]["en-us"])
	text = _SCI_RE.sub(rf"\1{sci_words}\2", text)
	to_word = _CLN["to_words"].get(lang, _CLN["to_words"]["en-us"])
	text = _RANGE_RE.sub(rf"\1{to_word}\2", text)
	point_word = _CLN["point_words"].get(lang, _CLN["point_words"]["en-us"])

	def repl_dec(m):
		return f"{m.group(1)}{point_word}{' '.join(list(m.group(2)))}"

	return _DECIMAL_RE.sub(repl_dec, text)


def expand_abbreviations(text):
	for regex, replacement in _abbreviations:

		def match_case(m, replacement=replacement):
			orig = m.group(0)
			if orig.isupper(): return replacement.upper()
			elif orig.istitle() or (len(orig) > 0 and orig[0].isupper()): return replacement.capitalize()
			return replacement

		text = re.sub(regex, match_case, text)
	return _INITIALS_RE.sub(r"\1\2", text)


def advanced_text_cleaning(text):
	text = re.sub(r'[“”«»‟"]', '"', text)
	text = re.sub(r"[‘’`´']", "'", text)
	text = re.sub(r'[()\[\]]', ", ", text)
	text = re.sub(r'["`]', "", text)
	text = re.sub(r"\s+([.,!?:;])", r"\1", text)
	text = re.sub(r"(,\s*)+", ", ", text)
	text = re.sub(r"([.,!?:;])(?=[a-zA-Z])", r"\1 ", text)
	text = re.sub(r"\s*—\s*", "—", text)
	return re.sub(r"\s+", " ", text.strip())


def _canon(lang):
	return _ALIASES.get(lang, lang)


def _g2p(text, lang):
	lang = _canon(lang)
	if lang == "ja": return japanese_to_ipa3(text)
	if lang == "ru":
		phonemes = espeak_ipa(text, "ru")
		return phonemes if phonemes else text
	if lang == "en-us": return espeak_ipa(text, "en-us")
	if lang in _VOICES: return espeak_ipa(text, lang)
	return text


def text_cleaners(text, language="en-us", language_map=None):
	if language_map is None: language_map = _LANG_MAP
	lang = language_map.get(language, language)
	text = text.strip()
	text = clean_urls(text)
	text = clean_punctuation(text)
	text = expand_abbreviations(text)
	text = replace_roman_numerals(text)
	text = remove_number_commas(text)
	text = process_currencies(text, lang)
	text = expand_numbers_and_symbols(text, lang)
	text = advanced_text_cleaning(text)
	return _g2p(text, lang).strip()


def split_sentences(text):
	text = text.strip()
	text = re.sub(r"[‘’`´]", "'", text)
	text = re.sub(r"[“”«»‟]", '"', text)
	text = expand_abbreviations(text)
	text = clean_punctuation(text)
	text = re.sub(r"\n+", ". ", text)
	text = re.sub(r'[()\[\]]', ", ", text)
	text = re.sub(r'["`]', "", text)
	text = re.sub(r"\s+([.,!?:;])", r"\1", text)
	text = re.sub(r"(,\s*)+", ", ", text)
	text = re.sub(r"\s+", " ", text.strip())
	sentences = re.split(r'([.!?]+)(?:\s+|$)', text)
	result = []
	for i in range(0, len(sentences) - 1, 2):
		sentence = sentences[i].strip()
		if sentence: result.append(f"{sentence}{sentences[i + 1]}")
	if len(sentences) % 2 == 1 and sentences[-1].strip():
		result.append(sentences[-1].strip())
	return result


def combine_sentences(sentences, max_length=300):
	if not sentences: return []
	combined, current_chunk, current_length = [], [], 0
	for sentence in sentences:
		sentence_length = len(sentence)
		if current_length + sentence_length + (1 if current_chunk else 0) > max_length:
			if current_chunk: combined.append(" ".join(current_chunk))
			current_chunk, current_length = [sentence], sentence_length
		else:
			current_chunk.append(sentence)
			current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)
	if current_chunk: combined.append(" ".join(current_chunk))
	return combined


def split_by_language(text, default_language="en-us"):
	words = re.findall(r"(?:\w+(?:[-'.,]\w+)*)|[^\w\s]", text)
	segments, current_segment = [], []
	current_language = default_language
	for word in words:
		if re.match(r"^[^\w]$", word):
			current_segment.append(word)
			continue
		word_language = detect_language_for_word(word)
		if word_language != current_language and current_segment:
			segments.append({"text": " ".join(current_segment), "language": current_language})
			current_segment, current_language = [word], word_language
		else:
			current_segment.append(word)
			if word_language != default_language: current_language = word_language
	if current_segment: segments.append({"text": " ".join(current_segment), "language": current_language})
	return segments


def split_and_process_text(text, language="en-us", max_length=300, combine=True, language_map=None):
	if language_map is None: language_map = _LANG_MAP
	sentences = split_sentences(text)
	if combine and len(sentences) > 1: sentences = combine_sentences(sentences, max_length)
	result = []
	for sentence in sentences:
		for segment in split_by_language(sentence, default_language=language):
			if segment["text"].strip(): result.append(segment)
	processed = []
	for segment in result:
		cleaned = text_cleaners(segment["text"], segment["language"], language_map)
		if cleaned.strip(): processed.append({"text": cleaned, "language": segment["language"]})
	return processed


def text_to_sequence(text, language="en-us", language_map=None):
	if has_tags(text):  # keep brackets through G2P, then events→symbols / styles stripped for the id table
		phones = text_to_phonemes_tagged(text, language, language_map)
		return [_symbol_to_id[s] for s in materialize_tags_for_ids(phones) if s in _symbol_to_id]
	return [_symbol_to_id[s] for s in text_cleaners(text, language, language_map) if s in _symbol_to_id]


def cleaned_text_to_sequence(cleaned_text):
	text = materialize_tags_for_ids(cleaned_text) if has_tags(cleaned_text) else fold_phones(cleaned_text)  # IPA filelists may carry `[neutral]`/`[laugh]`; strip/expand before 177 lookup
	return [_symbol_to_id[s] for s in text if s in _symbol_to_id]


def sequence_to_text(sequence):
	return "".join(_id_to_symbol[sid] for sid in sequence)


def combine_chunks(filepaths_and_text, max_length=300):
	combined, current_chunk, current_length = [], [], 0
	for item in filepaths_and_text:
		text_length = len(item[-1])
		if current_length + text_length + 1 <= max_length:
			current_chunk.append(item)
			current_length += text_length + (1 if current_chunk else 0)
		else:
			if current_chunk:
				combined.append(current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)])
			current_chunk, current_length = [item], text_length
	if current_chunk: combined.append(current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)])
	return combined


def text_to_phonemes_raw(text, language="en-us", language_map=None):
	"""Orthography → IPA. Bracket tags are preserved (`[neutral] … [laugh] …`)."""
	if language_map is None:
		language_map = _LANG_MAP
	return text_to_phonemes_tagged(text, language, language_map)


def preprocess_filelists(filelists, language="en-us", combine_text=True, language_map=None):
	if language_map is None: language_map = _LANG_MAP
	for filelist in filelists:
		filepaths_and_text = load_filepaths_and_text(filelist)
		for i, item in enumerate(filepaths_and_text):
			lang = item[2] if len(item) >= 4 else language  # per-row lang when path|spk|lang|text
			filepaths_and_text[i][-1] = text_to_phonemes_raw(item[-1], lang, language_map)
		if combine_text: filepaths_and_text = combine_chunks(filepaths_and_text)
		with open(f"{filelist}.cleaned", "w", encoding="utf-8") as f:
			f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])
