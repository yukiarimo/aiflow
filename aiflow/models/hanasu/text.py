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
_abbreviations = [(re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1]) for x in [("mrs", "misess"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("drs", "doctors"), ("u.s.", "US"), ("u.n.", "UN"), ("e.g", "for example"), ("i.e", "that is"), ("etc", "et cetera"), ("d.r.", "doctor")]]
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def normalize_numbers(text):
    text = re.sub(r'(\d),(\d)', r'\1\2', text) # Remove commas from numbers (123,678 -> 123678) - only when surrounded by digits
    text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text) # Convert decimals to spoken form (4.2 -> 4 point 2) - only when surrounded by digits
    return text

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:  text = re.sub(regex, replacement, text)
    return text

def text_cleaners(text=None, language="en-us"):
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
    """Split text into sentences with cleaning and normalization"""
    # Handle newlines that act as sentence boundaries (add comma before converting to space)
    text = re.sub(r'([^\.\!\?])\n+([A-Z])', r'\1, \2', text)

    # Replace remaining newlines and normalize whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())

    # Handle em dashes and en dashes (with spaces around them) - convert to comma
    text = re.sub(r'\s+[–—]\s+', ', ', text)

    # Handle hyphens within words - convert to space
    text = re.sub(r'(\w)-(\w)', r'\1 \2', text)

    # Handle remaining standalone dashes, semicolons, colons as commas
    text = re.sub(r'[;:]', ',', text)

    # Replace … with period
    text = re.sub(r'…', '.', text)

    text = re.sub(r'[()""''"`]', '', text)
    sentences = re.split(r'([.!?])', text)

    # Reconstruct sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
            result.append(sentence + punctuation)

    # Handle case where text doesn't end with punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip(): result.append(sentences[-1].strip())

    return result

def combine_sentences(sentences, max_length=300):
    """Combine sentences to stay under max_length characters"""
    if not sentences:  return []
    combined = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed max_length
        if current_length + sentence_length + (1 if current_chunk else 0) > max_length:
            # Save current chunk if it exists
            if current_chunk: combined.append(" ".join(current_chunk))
            # Start new chunk with current sentence
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + (1 if len(current_chunk) > 1 else 0)

    # Add final chunk if it exists
    if current_chunk: combined.append(" ".join(current_chunk))

    return combined

def split_and_process_text(text, language="en-us", max_length=300, combine=True):
    """
    Split text into sentences, optionally combine them, and process for TTS

    Args:
        text: Input text to process
        language: Language for phonemization
        max_length: Maximum character length for combined chunks
        combine: Whether to combine sentences under max_length

    Returns:
        List of processed text chunks ready for TTS
    """
    # Split into sentences
    sentences = split_sentences(text)

    # Optionally combine sentences
    if combine and len(sentences) > 1: sentences = combine_sentences(sentences, max_length)

    # Process each chunk through text cleaners
    processed_chunks = []
    for chunk in sentences:
        if chunk.strip():  # Only process non-empty chunks
            processed_chunk = text_cleaners(chunk, language)
            processed_chunks.append(processed_chunk)

    return processed_chunks

def symbol_to_id_converter(symbol):
    if symbol in _symbol_to_id:
        return _symbol_to_id[symbol]
    raise ValueError(f"Symbol '{symbol}' not found in the symbol table.")

def text_to_sequence(text=None, language="en-us"):
    clean_text = text_cleaners(text, language)
    return [_symbol_to_id[s] for s in clean_text if s in _symbol_to_id]

def cleaned_text_to_sequence(cleaned_text): return [_symbol_to_id[s] for s in cleaned_text if s in _symbol_to_id]

def sequence_to_text(sequence): return "".join(_id_to_symbol[sid] for sid in sequence)

def combine_chunks(filepaths_and_text, max_length=300):
    """Combine text chunks to stay under max_length characters"""
    combined = []
    current_chunk = []
    current_length = 0

    for item in filepaths_and_text:
        text = item[-1]  # Get text (last element)
        text_length = len(text)

        if current_length + text_length + 1 <= max_length:  # +1 for space
            current_chunk.append(item)
            current_length += text_length + (1 if current_chunk else 0)
        else:
            if current_chunk:
                # Combine current chunk
                combined_item = current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)]
                combined.append(combined_item)
            current_chunk = [item]
            current_length = text_length

    if current_chunk:
        combined_item = current_chunk[0][:-1] + [" ".join(c[-1] for c in current_chunk)]
        combined.append(combined_item)

    return combined

def preprocess_filelists(filelists, language="en-us", combine_text=True):
    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)

        # Clean text
        for i, item in enumerate(filepaths_and_text):
            original_text = item[-1]
            print(f"Processing: {original_text}")
            filepaths_and_text[i][-1] = text_cleaners(original_text, language)

        # Combine chunks if requested
        if combine_text: filepaths_and_text = combine_chunks(filepaths_and_text)

        new_filelist = filelist + ".cleaned"
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])