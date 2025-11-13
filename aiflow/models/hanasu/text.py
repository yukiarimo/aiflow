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
    # Add custom ones here, e.g., (re.compile(r"\bcustom\.", re.IGNORECASE), "custom expansion")
    (re.compile(r"\bvs\.", re.IGNORECASE), "versus"),
    (re.compile(r"\binc\.", re.IGNORECASE), "incorporated")
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
    # Uniform quotes: Replace curly with straight
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    text = re.sub(r'[«»]', '"', text)
    text = re.sub(r'[‟]', '"', text)
    # Remove quotes, parentheses, and brackets
    text = re.sub(r'[()""''`]', '', text)
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

def text_cleaners(text, language="en-us"):
    text = normalize_numbers(expand_abbreviations(text))
    text = advanced_text_cleaning(text)  # New step for better cleaning
    print(f"Cleaned text before phonemization: {text}")
    if language == "en-us":
        phonemes = backend.phonemize([text], strip=False)[0]
    elif language == "ru":
        result = ru_backend.phonemize([text], strip=False)
        phonemes = result[0] if result and result[0] else ""
        if not phonemes:
            print(f"Warning: phonemizer returned empty result for: {text}")
    elif language == "ja":
        phonemes = ja_backend.romaji(text)
    else:
        phonemes = text
    return phonemes

def split_sentences(text):
    """Split text into sentences with cleaning and normalization."""
    # Expand abbreviations FIRST before any other processing
    text = expand_abbreviations(text)
    
    text = re.sub(r'\n+', '. ', text)  # All newlines -> period (not comma)
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'[;:]', ',', text)  # Semicolons/colons -> comma
    text = re.sub(r'…', '.', text)  # Ellipsis -> period
    text = re.sub(r'[()""''"`]', '', text)  # Remove quotes/parentheses
    text = re.sub(r'\.\.\.', ', ', text)  # Convert three dots to comma before splitting
    
    # Remove duplicate punctuation and clean up
    text = re.sub(r'([.!?,])\s*\1+', r'\1', text)  # Remove duplicate punctuation
    text = re.sub(r',\s*([.!?])', r'\1', text)  # Remove comma before sentence-ending punctuation
    text = re.sub(r'([.!?])\s*,', r'\1', text)  # Remove comma after sentence-ending punctuation
    text = re.sub(r'\s+', ' ', text.strip())  # Clean spaces again

    sentences = re.split(r'([.!?])', text)
    result = []
    
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:  # Only add non-empty sentences
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

def split_and_process_text(text, language="en-us", max_length=300, combine=True):
    """Split, optionally combine, and process text for TTS."""
    sentences = split_sentences(text)
    if combine and len(sentences) > 1: sentences = combine_sentences(sentences, max_length)
    return [text_cleaners(chunk, language) for chunk in sentences if chunk.strip()]

def text_to_sequence(text, language="en-us"): return [_symbol_to_id[s] for s in text_cleaners(text, language) if s in _symbol_to_id]
def cleaned_text_to_sequence(cleaned_text): return [_symbol_to_id[s] for s in cleaned_text if s in _symbol_to_id]
def sequence_to_text(sequence): return "".join(_id_to_symbol[sid] for sid in sequence)

def combine_chunks(filepaths_and_text, max_length=300):
    """Combine text chunks from a filelist to stay under max_length."""
    combined, current_chunk, current_length = [], [], 0
    for item in filepaths_and_text:
        text, text_length = item[-1], len(item[-1]) # Text is the last element
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

def preprocess_filelists(filelists, language="en-us", combine_text=True):
    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)

        for i, item in enumerate(filepaths_and_text): # Clean text in-place
            print(f"Processing: {item[-1]}")
            filepaths_and_text[i][-1] = text_cleaners(item[-1], language)

        if combine_text: filepaths_and_text = combine_chunks(filepaths_and_text)
        new_filelist = f"{filelist}.cleaned"
        with open(new_filelist, "w", encoding="utf-8") as f: f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])

def text_to_phonemes_raw(text, language="en-us"):
    """Convert text directly to phonemes without any preprocessing/cleaning."""
    if language == "en-us":
        phonemes = backend.phonemize([text], strip=False)[0]
    elif language == "ru":
        result = ru_backend.phonemize([text], strip=False)
        phonemes = result[0] if result and result[0] else ""
        if not phonemes:
            print(f"Warning: phonemizer returned empty result for: {text}")
    elif language == "ja":
        phonemes = ja_backend.romaji(text)
    else:
        phonemes = text
    return phonemes

def preprocess_filelists_raw(filelists, language="en-us", combine_text=True):
    """Preprocess filelists without text cleaning - raw phonemization only."""
    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)
        print(f"Original lines: {len(filepaths_and_text)}")

        for i, item in enumerate(filepaths_and_text):
            print(f"Processing raw: {item[-1]}")
            filepaths_and_text[i][-1] = text_to_phonemes_raw(item[-1], language)

        if combine_text: 
            filepaths_and_text = combine_chunks(filepaths_and_text)
            print(f"After combining: {len(filepaths_and_text)}")
        
        new_filelist = f"{filelist}.cleaned"
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines([f"{'|'.join(x)}\n" for x in filepaths_and_text])