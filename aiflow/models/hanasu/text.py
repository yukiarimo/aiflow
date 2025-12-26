import re
from phonemizer.backend import EspeakBackend
import cutlet
from .utils import load_filepaths_and_text

backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True, words_mismatch="warn")
ru_backend = EspeakBackend("ru", preserve_punctuation=True, with_stress=True, words_mismatch="warn")
ja_backend = cutlet.Cutlet(use_foreign_spelling=False)

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
SPACE_ID = symbols.index(" ")
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Russian alphabet (Cyrillic)
RUSSIAN_ALPHABET = set("абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

# Japanese hiragana and katakana
JAPANESE_HIRAGANA = set("あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん")
JAPANESE_KATAKANA = set("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン")
JAPANESE_KANJI_RANGES = [
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
]

def is_japanese(char):
    """Check if character is Japanese."""
    if char in JAPANESE_HIRAGANA or char in JAPANESE_KATAKANA:
        return True
    code = ord(char)
    for start, end in JAPANESE_KANJI_RANGES:
        if start <= code <= end:
            return True
    return False

def is_russian(char):
    """Check if character is Russian (Cyrillic)."""
    return char in RUSSIAN_ALPHABET

def detect_language_for_word(word):
    """Detect the primary language of a word."""
    # Count characters by language
    russian_count = sum(1 for c in word if is_russian(c))
    japanese_count = sum(1 for c in word if is_japanese(c))

    if russian_count > 0 and russian_count >= japanese_count:
        return "ru"
    elif japanese_count > 0:
        return "ja"
    else:
        return "en-us"

# Expanded abbreviations list (add more as needed)
_abbreviations = [
    (re.compile(r"\bdr\.", re.IGNORECASE), "doctor"),
    (re.compile(r"\bmr\.", re.IGNORECASE), "mister"),
    (re.compile(r"\bmrs\.", re.IGNORECASE), "misess"),
    (re.compile(r"\bms\.", re.IGNORECASE), "miss"),
    (re.compile(r"\bst\.", re.IGNORECASE), "saint"),
    (re.compile(r"\bdrs\.", re.IGNORECASE), "doctors"),
    (re.compile(r"\bu\.s\.a\.?", re.IGNORECASE), "USA"),
    (re.compile(r"\bu\.s\.s\.r\.?", re.IGNORECASE), "USSR"),
    (re.compile(r"\bu\.s\.", re.IGNORECASE), "US"),
    (re.compile(r"\bu\.n\.", re.IGNORECASE), "UN"),
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.", re.IGNORECASE), "etcetera"),
    (re.compile(r"\ba\.s\.m\.r\.?", re.IGNORECASE), "ASMR"),
    (re.compile(r"\ba\.m\.", re.IGNORECASE), "AM"),
    (re.compile(r"\bp\.m\.", re.IGNORECASE), "PM"),
    (re.compile(r"\bd\.r\.", re.IGNORECASE), "doctor"),
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\binc\.", re.IGNORECASE), "incorporated"),
    (re.compile(r"\bain't\b", re.IGNORECASE), "am not"),
]

def normalize_numbers(text):
    text = re.sub(r'(?<=\d),(?=\d)', '', text)  # Remove commas in numbers
    text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text)  # 4.2 -> 4 point 2
    return text

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def advanced_text_cleaning(text):
    """Additional cleaning before phonemization."""
    # Normalize various double-quote like characters to straight "
    text = re.sub(r'[“”«»‟"]', '"', text)
    # Normalize various single-quote like characters to straight '
    text = re.sub(r"[‘’`´']", "'", text)
    # Remove quotes, parentheses, and brackets, BUT KEEP APOSTROPHE
    text = re.sub(r'[()\[\]"`]', '', text)
    # Convert ellipses to comma for better flow
    text = re.sub(r'\.\.\.', ', ', text)
    # Normalize semicolons and colons to commas
    text = re.sub(r'[;:]', ',', text)
    # Remove multiple commas and weird punctuation doubles
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\. +\.', '.', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*\?\s*', '? ', text)
    text = re.sub(r'\s*!\s*', '! ', text)
    # Remove extra spaces and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Handle times (e.g., 1:35 -> 1 35)
    text = re.sub(r'(\d):(\d)', r'\1 \2', text)
    # Remove trailing commas before punctuation
    text = re.sub(r',\s*([.!?])', r'\1', text)
    # Replace all ". ," with just ". "
    text = re.sub(r'\.\s*,', '. ', text)
    # Replace ", ," with just ", "
    text = re.sub(r',\s*,', ', ', text)
    # Replace "! ," with just "! "
    text = re.sub(r'!\s*,', '! ', text)
    # Replace "? ," with just "? "
    text = re.sub(r'\?\s*,', '? ', text)
    # Replace em dashes with ", "
    text = re.sub(r'—', ', ', text)
    # Clean double spaces again
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def text_cleaners(text, language="en-us", language_map=None):
    if language_map is None:
        language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

    text = normalize_numbers(expand_abbreviations(text))
    text = advanced_text_cleaning(text)
    print(f"Cleaned text before phonemization: {text}")

    # Map the language code
    phonemizer_lang = language_map.get(language, language)

    if phonemizer_lang == "en-us":
        phonemes = backend.phonemize([text], strip=False)[0]
    elif phonemizer_lang == "ru":
        result = ru_backend.phonemize([text], strip=False)
        phonemes = result[0] if result and result[0] else ""
        if not phonemes:
            print(f"Warning: phonemizer returned empty result for: {text}")
    elif phonemizer_lang == "ja":
        phonemes = ja_backend.romaji(text)
    else:
        phonemes = text
    return phonemes

def split_sentences(text):
    """Split text into sentences with cleaning and normalization."""
    # 1. Normalize smart quotes and dashes EARLY
    text = re.sub(r"[‘’`´]", "'", text) # not fixing ‘
    text = re.sub(r"[“”«»‟]", '"', text)

    # Expand abbreviations
    text = expand_abbreviations(text)

    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[;:]', ',', text)
    text = re.sub(r'…', '.', text)

    # Remove quotes/parentheses/brackets safely, BUT KEEP APOSTROPHE AND HYPHEN
    text = re.sub(r'[()\[\]"]', '', text)
    text = re.sub(r'\.\.\.', ', ', text)

    # Remove duplicate punctuation and clean up
    text = re.sub(r'([.!?,])\s*\1+', r'\1', text)
    text = re.sub(r',\s*([.!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*,', r'\1', text)
    text = re.sub(r'\s+', ' ', text.strip())

    sentences = re.split(r'([.!?])', text)
    result = []

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:
            result.append(f"{sentence}{sentences[i+1]}")

    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())

    return result

def combine_sentences(sentences, max_length=300):
    """Combine sentences to stay under max_length characters."""
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
    """Split text into segments by language, preserving punctuation."""
    # REGEX UPDATE:
    # Matches words, including those with hyphens or apostrophes inside (e.g., "jerry-rigged", "don't")
    words = re.findall(r"(?:\w+(?:[-']\w+)*)|[^\w\s]", text)

    segments = []
    current_segment = []
    current_language = default_language

    i = 0
    while i < len(words):
        word = words[i]

        # Skip pure punctuation for language detection
        if re.match(r'^[^\w]$', word):
            current_segment.append(word)
            i += 1
            continue

        # Detect language of this word
        word_language = detect_language_for_word(word)

        if word_language != current_language and current_segment:
            segments.append({
                'text': ' '.join(current_segment),
                'language': current_language
            })
            current_segment = [word]
            current_language = word_language
        else:
            current_segment.append(word)
            if word_language != default_language:
                current_language = word_language

        i += 1

    if current_segment:
        segments.append({
            'text': ' '.join(current_segment),
            'language': current_language
        })

    return segments

def split_and_process_text(text, language="en-us", max_length=300, combine=True, language_map=None):
    """Split, optionally combine, and process text for TTS with multilingual support."""
    print(f"[DEBUG] Starting split_and_process_text with text length: {len(text)}")
    if language_map is None:
        language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

    print(f"[DEBUG] Splitting sentences...")
    sentences = split_sentences(text)
    print(f"[DEBUG] Got {len(sentences)} sentences")

    if combine and len(sentences) > 1:
        print(f"[DEBUG] Combining sentences...")
        sentences = combine_sentences(sentences, max_length)
        print(f"[DEBUG] After combining: {len(sentences)} chunks")

    print(f"[DEBUG] Splitting by language...")
    result = []
    for i, sentence in enumerate(sentences):
        print(f"[DEBUG] Processing sentence {i+1}/{len(sentences)}")
        lang_segments = split_by_language(sentence, default_language=language)
        for segment in lang_segments:
            if segment['text'].strip():
                result.append({
                    'text': segment['text'],
                    'language': segment['language']
                })

    print(f"[DEBUG] Processing {len(result)} segments with phonemizer...")
    processed = []
    for j, segment in enumerate(result):
        print(f"[DEBUG] Phonemizing segment {j+1}/{len(result)}: {segment['text'][:50]}...")
        cleaned = text_cleaners(segment['text'], segment['language'], language_map)
        if cleaned.strip():
            processed.append({
                'text': cleaned,
                'language': segment['language']
            })

    print(f"[DEBUG] Done! Returning {len(processed)} processed segments")
    return processed

def text_to_sequence(text, language="en-us", language_map=None):
    return [_symbol_to_id[s] for s in text_cleaners(text, language, language_map) if s in _symbol_to_id]

def cleaned_text_to_sequence(cleaned_text):
    return [_symbol_to_id[s] for s in cleaned_text if s in _symbol_to_id]

def sequence_to_text(sequence):
    return "".join(_id_to_symbol[sid] for sid in sequence)

def combine_chunks(filepaths_and_text, max_length=300):
    """Combine text chunks from a filelist to stay under max_length."""
    combined, current_chunk, current_length = [], [], 0
    for item in filepaths_and_text:
        text, text_length = item[-1], len(item[-1])
        if current_length + text_length + 1 <= max_length:
            current_chunk.append(item)
            current_length += text_length + (1 if current_chunk else 0)
        else:
            if current_chunk:
                combined_item = current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)]
                combined.append(combined_item)
            current_chunk, current_length = [item], text_length

    if current_chunk:
        combined_item = current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)]
        combined.append(combined_item)
    return combined

def preprocess_filelists(filelists, language="en-us", combine_text=True, language_map=None):
    if language_map is None:
        language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)

        for i, item in enumerate(filepaths_and_text):
            print(f"Processing: {item[-1]}")
            filepaths_and_text[i][-1] = text_cleaners(item[-1], language, language_map)

        if combine_text:
            filepaths_and_text = combine_chunks(filepaths_and_text)
        new_filelist = f"{filelist}.cleaned"
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])

def text_to_phonemes_raw(text, language="en-us", language_map=None):
    """Convert text directly to phonemes without any preprocessing/cleaning."""
    if language_map is None:
        language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

    phonemizer_lang = language_map.get(language, language)

    if phonemizer_lang == "en-us":
        phonemes = backend.phonemize([text], strip=False)[0]
    elif phonemizer_lang == "ru":
        result = ru_backend.phonemize([text], strip=False)
        phonemes = result[0] if result and result[0] else ""
        if not phonemes:
            print(f"Warning: phonemizer returned empty result for: {text}")
    elif phonemizer_lang == "ja":
        phonemes = ja_backend.romaji(text)
    else:
        phonemes = text
    return phonemes

def preprocess_filelists_raw(filelists, language="en-us", combine_text=True, language_map=None):
    """Preprocess filelists without text cleaning - raw phonemization only."""
    if language_map is None:
        language_map = {"en-us": "en-us", "ru": "ru", "ja": "ja"}

    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)
        print(f"Original lines: {len(filepaths_and_text)}")

        for i, item in enumerate(filepaths_and_text):
            print(f"Processing raw: {item[-1]}")
            filepaths_and_text[i][-1] = text_to_phonemes_raw(item[-1], language, language_map)

        if combine_text:
            filepaths_and_text = combine_chunks(filepaths_and_text)
            print(f"After combining: {len(filepaths_and_text)}")

        new_filelist = f"{filelist}.cleaned"
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])