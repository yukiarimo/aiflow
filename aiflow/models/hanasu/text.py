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
_abbreviations = [(re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1]) for x in [("mrs.", "misess"), ("mr.", "mister"), ("dr.", "doctor"), ("st.", "saint"), ("drs.", "doctors"), ("u.s.", "US"), ("u.n.", "UN"), ("e.g.", "for example"), ("i.e.", "that is"), ("etc.", "et cetera"), ("d.r.", "doctor")]]

def normalize_numbers(text):
    text = re.sub(r'(\d),(\d)', r'\1\2', text) # 123,678 -> 123678
    text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text) # 4.2 -> 4 point 2
    return text

def expand_abbreviations(text):
    for regex, replacement in _abbreviations: text = re.sub(regex, replacement, text)
    return text

def text_cleaners(text, language="en-us"):
    text = normalize_numbers(expand_abbreviations(text))
    if language == "en-us": phonemes = backend.phonemize([text], strip=False)[0]
    elif language == "ru":
        result = ru_backend.phonemize([text], strip=False)
        phonemes = result[0] if result and result[0] else ""
        if not phonemes: print(f"Warning: phonemizer returned empty result for: {text}")
    elif language == "ja": phonemes = ja_backend.romaji(text)
    else: phonemes = text
    return phonemes

def split_sentences(text):
    """Split text into sentences with cleaning and normalization."""
    text = re.sub(r'([^\.\!\?])\n+([A-Z])', r'\1, \2', text) # Newlines mid-sentence -> comma
    text = re.sub(r'\n+', ' ', text) # Remaining newlines -> space
    text = re.sub(r'\s+', ' ', text.strip()) # Normalize whitespace
    text = re.sub(r'\s+[–—]\s+', ', ', text) # Em/en dashes -> comma
    text = re.sub(r'(\w)-(\w)', r'\1 \2', text) # Hyphens -> space
    text = re.sub(r'[;:]', ',', text) # Semicolons/colons -> comma
    text = re.sub(r'…', '.', text) # Ellipsis -> period
    text = re.sub(r'[()""''"`]', '', text) # Remove quotes/parentheses

    sentences = re.split(r'([.!?])', text)
    result = [f"{sentences[i].strip()}{sentences[i+1]}" for i in range(0, len(sentences) - 1, 2) if sentences[i].strip()]

    if len(sentences) % 2 == 1 and sentences[-1].strip(): result.append(sentences[-1].strip()) # Handle text that doesn't end with punctuation

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