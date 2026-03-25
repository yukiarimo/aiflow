import re
from phonemizer.backend import EspeakBackend
from unidecode import unidecode
import pyopenjtalk
from .utils import load_filepaths_and_text

backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True, words_mismatch="warn", language_switch="remove-flags")
ru_backend = EspeakBackend("ru", preserve_punctuation=True, with_stress=True, words_mismatch="warn")

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
SPACE_ID = symbols.index(" ")
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Language Detection Alphabets
RUSSIAN_ALPHABET = set("абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
JAPANESE_HIRAGANA = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
JAPANESE_KATAKANA = set("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")
JAPANESE_KANJI_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]

# Cleaning rules
_abbreviations = [(re.compile(r"\bdr\.", re.IGNORECASE), "Doctor"), (re.compile(r"\bmr\.", re.IGNORECASE), "Mister"), (re.compile(r"\bmrs\.", re.IGNORECASE), "Missus"), (re.compile(r"\bms\.", re.IGNORECASE), "Miss"), (re.compile(r"\bst\.", re.IGNORECASE), "Saint"), (re.compile(r"\bdrs\.", re.IGNORECASE), "Doctors"), (re.compile(r"\bprof\.", re.IGNORECASE), "Professor"), (re.compile(r"\bjr\.", re.IGNORECASE), "Junior"), (re.compile(r"\bsr\.", re.IGNORECASE), "Senior"), (re.compile(r"\bno\.", re.IGNORECASE), "number"), (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"), (re.compile(r"\bdept\.", re.IGNORECASE), "department"), (re.compile(r"\bu\.s\.a\.?", re.IGNORECASE), "USA"), (re.compile(r"\bu\.s\.s\.r\.?", re.IGNORECASE), "USSR"), (re.compile(r"\bu\.s\.", re.IGNORECASE), "US"), (re.compile(r"\bd\.c\.", re.IGNORECASE), "DC"), (re.compile(r"\bave\.", re.IGNORECASE), "Avenue"), (re.compile(r"\bblvd\.", re.IGNORECASE), "Boulevard"), (re.compile(r"\brd\.", re.IGNORECASE), "Road"), (re.compile(r"\bhwy\.", re.IGNORECASE), "Highway"), (re.compile(r"\bu\.n\.", re.IGNORECASE), "UN"), (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"), (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"), (re.compile(r"\betc\.", re.IGNORECASE), "etcetera"), (re.compile(r"\ba\.s\.m\.r\.?", re.IGNORECASE), "ASMR"), (re.compile(r"\ba\.m\.", re.IGNORECASE), "AM"), (re.compile(r"\bp\.m\.", re.IGNORECASE), "PM"), (re.compile(r"\bvs\.", re.IGNORECASE), "versus"), (re.compile(r"\binc\.", re.IGNORECASE), "Incorporated"), (re.compile(r"\bain't\b", re.IGNORECASE), "am not")]
CURRENCY_MAP = {'$': {'en-us': ('dollar', 'dollars'), 'ru': ('доллар', 'доллара', 'долларов'), 'ja': 'doru'}, '€': {'en-us': ('euro', 'euros'), 'ru': ('евро', 'евро', 'евро'), 'ja': 'yuro'}, '£': {'en-us': ('pound', 'pounds'), 'ru': ('фунт', 'фунта', 'фунтов'), 'ja': 'pondo'}, '¥': {'en-us': ('yen', 'yen'), 'ru': ('иен', 'иен', 'иен'), 'ja': 'en'}, '₽': {'en-us': ('ruble', 'rubles'), 'ru': ('рубль', 'рубля', 'рублей'), 'ja': 'ruuburu'}}
SUFFIX_MAP = {'K': {'en-us': 'thousand', 'ru': 'тысяч', 'ja': 'sen'}, 'M': {'en-us': 'million', 'ru': 'миллионов', 'ja': 'mirion'}, 'B': {'en-us': 'billion', 'ru': 'миллиардов', 'ja': 'birion'}, 'T': {'en-us': 'trillion', 'ru': 'триллионов', 'ja': 'toririon'}}

# Japanese Pyopenjtalk Processing
_japanese_characters = re.compile(r'[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')
_japanese_marks = re.compile(r'[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]')
_symbols_to_japanese = [(re.compile('%s' % x[0]), x[1]) for x in [('％', 'パーセント')]]
_romaji_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [('u', 'ɯ'), ('ʧ', 'tʃ'), ('j', 'dʑ'), ('y', 'j'), ('ni', 'n^i'), ('nj', 'n^'), ('hi', 'çi'), ('hj', 'ç'), ('f', 'ɸ'), ('I', 'i*'), ('U', 'ɯ*'), ('r', 'ɾ')]]
_real_sokuon = [(re.compile('%s' % x[0]), x[1]) for x in [(r'Q([↑↓]*[kg])', r'k#\1'), (r'Q([↑↓]*[tdjʧ])', r't#\1'), (r'Q([↑↓]*[sʃ])', r's\1'), (r'Q([↑↓]*[pb])', r'p#\1')]]
_real_hatsuon = [(re.compile('%s' % x[0]), x[1]) for x in [(r'N([↑↓]*[pbm])', r'm\1'), (r'N([↑↓]*[ʧʥj])', r'n^\1'), (r'N([↑↓]*[tdn])', r'n\1'), (r'N([↑↓]*[kg])', r'ŋ\1')]]


def symbols_to_japanese(text):
	for regex, replacement in _symbols_to_japanese:
		text = re.sub(regex, replacement, text)
	return text


def japanese_to_romaji_with_accent(text):
	text = symbols_to_japanese(text)
	sentences = re.split(_japanese_marks, text)
	marks = re.findall(_japanese_marks, text)
	out_text = ''

	for i, sentence in enumerate(sentences):
		if re.match(_japanese_characters, sentence):
			if out_text != '': out_text += ' '
			labels = pyopenjtalk.extract_fullcontext(sentence)

			for n, label in enumerate(labels):
				phoneme = re.search(r'\-([^\+]*)\+', label).group(1)

				if phoneme not in ['sil', 'pau']:
					out_text += phoneme.replace('ch', 'ʧ').replace('sh', 'ʃ').replace('cl', 'Q')
				else:
					continue

				a1 = int(re.search(r"/A:(\-?[0-9]+)\+", label).group(1))
				a2 = int(re.search(r"\+(\d+)\+", label).group(1))
				a3 = int(re.search(r"\+(\d+)/", label).group(1))
				a2_next = -1 if re.search(r'\-([^\+]*)\+', labels[n + 1]).group(1) in ['sil', 'pau'] else int(re.search(r"\+(\d+)\+", labels[n + 1]).group(1))

				if a3 == 1 and a2_next == 1: out_text += ' '
				elif a1 == 0 and a2_next == a2 + 1: out_text += '↓'
				elif a2 == 1 and a2_next == 2: out_text += '↑'

		if i < len(marks):
			out_text += unidecode(marks[i]).replace(' ', '')
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
	text = japanese_to_romaji_with_accent(text).replace('...', '…')
	text = get_real_sokuon(text)
	text = get_real_hatsuon(text)

	for regex, replacement in _romaji_to_ipa:
		text = re.sub(regex, replacement, text)
	return text


def japanese_to_ipa3(text):
	text = japanese_to_ipa2(text).replace('n^', 'ȵ').replace('ʃ', 'ɕ').replace('*', '\u0325').replace('#', '\u031a')
	text = re.sub(r'([aiɯeo])\1+', lambda x: x.group(0)[0] + 'ː' * (len(x.group(0)) - 1), text)
	text = re.sub(r'((?:^|\s)(?:ts|tɕ|[kpt]))', r'\1ʰ', text)
	return text


def is_japanese(char):
	if char in JAPANESE_HIRAGANA or char in JAPANESE_KATAKANA: return True
	for start, end in JAPANESE_KANJI_RANGES:
		if start <= ord(char) <= end: return True
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
	return re.sub(r'https?://\S+|www\.\S+', '', text)


def clean_punctuation(text):
	text = re.sub(r'\s*\.\.\.\s*', '—', text)  # " ... " -> "—"
	text = re.sub(r'(?:\s*\.\s*){3,}', '—', text)  # " . . . " -> "—"
	text = re.sub(r'\?+!', '?', text)  # ?! -> ?
	text = re.sub(r'!\+?\?', '?', text)  # !? -> ?
	text = re.sub(r'\?+', '?', text)  # ?? -> ?
	text = re.sub(r'!+', '!', text)  # !! -> !
	text = re.sub(r'\s*–\s*', '-', text)  # en-dash to hyphen
	text = re.sub(r'\s+-\s+', '—', text)  # Isolated hyphen to em-dash (" - " -> "—")
	text = re.sub(r'\s*—\s*', '—', text)  # Remove spaces around em-dash
	text = re.sub(r'\*', '', text)  # Remove asterisks
	return text


def replace_roman_numerals(text):
	def roman_to_int(s):
		roman = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
		total = prev = 0
		for char in reversed(s.upper()):
			curr = roman[char]
			total += curr if curr >= prev else -curr
			prev = curr
		return total

	def repl(m):
		word = m.group(0)

		# Exclude ALL single letters to prevent changing names/initials
		if len(word) == 1:
			return word

		# Exclude historical/regional acronyms
		if word.upper() in ['DC', 'MD', 'CD', 'CV', 'IV', 'MC', 'DIV', 'BC', 'AD']:
			return word

		if re.match(r'^(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', word):
			return str(roman_to_int(word))
		return word

	# Only match uppercase Roman numerals, and NEVER if attached to a digit or apostrophe
	return re.sub(r"(?<![\d'])\b[MDCLXVI]+\b(?![\d'])", repl, text)


def remove_number_commas(text):
	"""Removes commas from numbers so the currency and suffix matchers don't break."""
	return re.sub(r'(?<=\d)\s*,\s*(?=\d)', '', text)


def process_currencies(text, lang):
	pattern = r'([$€£¥₽])\s*(\d+(?:\.\d+)?)\s*([KkMmBbTt]\b)?'
	pattern_rev = r'(\d+(?:\.\d+)?)\s*([KkMmBbTt]\b)?\s*([$€£¥₽])'

	def repl(m, is_rev=False):
		if not is_rev:
			sym, num_str, suf = m.groups()
		else:
			num_str, suf, sym = m.groups()

		val = float(num_str)
		c_info = CURRENCY_MAP[sym].get(lang, CURRENCY_MAP[sym]['en-us'])
		is_singular = (val == 1.0 and not suf)

		if lang == 'en-us':
			word = c_info[0] if is_singular else c_info[1]
		elif lang == 'ru':
			if suf: word = c_info[2]
			else:
				v, v1 = int(val) % 100, int(val) % 10
				if 11 <= v <= 19: word = c_info[2]
				elif v1 == 1: word = c_info[0]
				elif 2 <= v1 <= 4: word = c_info[1]
				else: word = c_info[2]
		else:
			word = c_info

		suf_word = ""
		if suf:
			suf_word = " " + SUFFIX_MAP[suf.upper()].get(lang, SUFFIX_MAP[suf.upper()]['en-us'])
		return f"{num_str}{suf_word} {word}"

	text = re.sub(pattern, lambda m: repl(m, False), text)
	text = re.sub(pattern_rev, lambda m: repl(m, True), text)
	return text


def expand_numbers_and_symbols(text, lang):
	def repl_suf(m):
		num, suf = m.group(1), m.group(2).upper()
		word = SUFFIX_MAP[suf].get(lang, SUFFIX_MAP[suf]['en-us'])
		return f"{num} {word}"

	text = re.sub(r'\b(\d+(?:\.\d+)?)\s*([KkMmBbTt])\b', repl_suf, text)
	sci_words = {"en-us": " times 10 to the ", "ru": " умножить на десять в степени ", "ja": " kakeru juu no "}.get(lang, " times 10 to the ")
	text = re.sub(r'(\d+(?:\.\d+)?)[eE]([+-]?\d+)', rf'\1{sci_words}\2', text)
	to_word = {"en-us": " to ", "ru": " до ", "ja": " kara "}.get(lang, " to ")
	text = re.sub(r'(\d+)\s*-\s*(\d+)', rf'\1{to_word}\2', text)
	point_word = {"en-us": " point ", "ru": " точка ", "ja": " ten "}.get(lang, " point ")

	def repl_dec(m):
		return f"{m.group(1)}{point_word}{' '.join(list(m.group(2)))}"

	text = re.sub(r'(\d+)\s*\.\s*(\d+)', repl_dec, text)
	return text


def expand_abbreviations(text):
	for regex, replacement in _abbreviations:

		def match_case(m):
			orig = m.group(0)
			if orig.isupper():
				return replacement.upper()
			elif orig.istitle() or (len(orig) > 0 and orig[0].isupper()):
				return replacement.capitalize()
			return replacement

		text = re.sub(regex, match_case, text)

	# Remove dots from single-letter initials (e.g., "Michael L. Bennett" -> "Michael L Bennett")
	text = re.sub(r'(^|\s)([A-Z])\.(?=\s|$|[.,;:!?])', r'\1\2', text)
	return text


def advanced_text_cleaning(text):
	text = re.sub(r'[“”«»‟"]', '"', text)
	text = re.sub(r"[‘’`´']", "'", text)

	# Replace formatting brackets with commas to preserve speech pacing
	text = re.sub(r'[()\[\]]', ", ", text)
	text = re.sub(r'["`]', "", text)

	# Safely remove spaces BEFORE punctuation
	text = re.sub(r"\s+([.,!?:;])", r"\1", text)

	# Collapse multiple commas (even if they have spaces between them)
	text = re.sub(r"(,\s*)+", ", ", text)

	# Safely ensure space AFTER punctuation (only if followed by a letter)
	text = re.sub(r"([.,!?:;])(?=[a-zA-Z])", r"\1 ", text)
	text = re.sub(r"\s*—\s*", "—", text)  # Protect em-dash spacing
	text = re.sub(r"\s+", " ", text.strip())
	return text


def text_cleaners(text, language="en-us", language_map=None):
	if language_map is None: language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}
	lang = language_map.get(language, language)

	# Strip any rogue spaces from the very beginning
	text = text.strip()
	text = clean_urls(text)
	text = clean_punctuation(text)
	text = expand_abbreviations(text)
	text = replace_roman_numerals(text)
	text = remove_number_commas(text)
	text = process_currencies(text, lang)
	text = expand_numbers_and_symbols(text, lang)
	text = advanced_text_cleaning(text)

	if lang == "en-us":
		phonemes = backend.phonemize([text], strip=False)[0]
	elif lang == "ru":
		result = ru_backend.phonemize([text], strip=False)
		phonemes = result[0] if result and result[0] else text
	elif lang == "ja":
		phonemes = japanese_to_ipa3(text)
	else:
		phonemes = text

	# Strip any rogue spaces from the very end of the phonemizer
	return phonemes.strip()


# PIPELINE & HELPERS
def split_sentences(text):
	text = text.strip()
	text = re.sub(r"[‘’`´]", "'", text)
	text = re.sub(r"[“”«»‟]", '"', text)

	text = expand_abbreviations(text)
	text = clean_punctuation(text)

	text = re.sub(r"\n+", ". ", text)

	# Replace formatting brackets with commas
	text = re.sub(r'[()\[\]]', ", ", text)
	text = re.sub(r'["`]', "", text)

	# Clean up spaces before punctuation and collapse double commas
	text = re.sub(r"\s+([.,!?:;])", r"\1", text)
	text = re.sub(r"(,\s*)+", ", ", text)
	text = re.sub(r"\s+", " ", text.strip())

	# Split on punctuation ONLY if followed by space or end of string
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
	if language_map is None: language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}
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
	return [_symbol_to_id[s] for s in text_cleaners(text, language, language_map) if s in _symbol_to_id]


def cleaned_text_to_sequence(cleaned_text):
	return [_symbol_to_id[s] for s in cleaned_text if s in _symbol_to_id]


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
	"""Raw bypass function for training pipelines."""
	if language_map is None: language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}
	lang = language_map.get(language, language)
	text = text.strip()

	if lang == "en-us":
		phonemes = backend.phonemize([text], strip=False)[0]
	elif lang == "ru":
		result = ru_backend.phonemize([text], strip=False)
		phonemes = result[0] if result and result[0] else text
	elif lang == "ja":
		phonemes = japanese_to_ipa3(text)
	else:
		phonemes = text
	return phonemes.strip()


def preprocess_filelists(filelists, language="en-us", combine_text=True, language_map=None):
	"""Preprocess filelists without text cleaning - raw phonemization only."""
	if language_map is None: language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

	for filelist in filelists:
		filepaths_and_text = load_filepaths_and_text(filelist)

		for i, item in enumerate(filepaths_and_text):
			filepaths_and_text[i][-1] = text_to_phonemes_raw(item[-1], language, language_map)

		if combine_text: filepaths_and_text = combine_chunks(filepaths_and_text)
		with open(f"{filelist}.cleaned", "w", encoding="utf-8") as f:
			f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])
