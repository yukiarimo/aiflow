import re
from phonemizer.backend import EspeakBackend
import cutlet
from .utils import load_filepaths_and_text
backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True, words_mismatch="warn")
ru_backend = EspeakBackend("ru", preserve_punctuation=True, with_stress=True, words_mismatch="warn")
ja_backend = cutlet.Cutlet(use_foreign_spelling=False)

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
SPACE_ID = symbols.index(" ")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("drs", "doctors"),
        ("u.s", "us"),
    ]
]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations: text = re.sub(regex, replacement, text)
    return text

def english_cleaners3(text=None, language="en-us"):
    if language == "en-us":
        phonemes = backend.phonemize([text], strip=False)[0]
    elif language == "ru":
        result = ru_backend.phonemize([text], strip=False)
        if not result or not result[0]:
            print(f"Warning: phonemizer returned empty result for: {text}")
            phonemes = ""
        else:
            phonemes = result[0]
    elif language == "ja":
        phonemes = ja_backend.romaji(text)
    else:
        phonemes = text
    return phonemes

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def symbol_to_id_converter(symbol):
    """Converts a symbol to its corresponding ID"""
    if symbol in _symbol_to_id.keys(): return _symbol_to_id[symbol]
    else: raise ValueError(f"Symbol '{symbol}' not found in the symbol table.")

def text_to_sequence(text=None, language="en-us"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    clean_text = english_cleaners3(text, language)
    for symbol in clean_text:
        if symbol in _symbol_to_id.keys():
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            continue
    return sequence

def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    for symbol in cleaned_text:
        if symbol in _symbol_to_id.keys():
            symbol_id = _symbol_to_id[symbol]
            sequence += [symbol_id]
        else: continue
    return sequence

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result

def preprocess_filelists(filelists, language="en-us"):
    for filelist in filelists:
        filepaths_and_text = load_filepaths_and_text(filelist)
        for i in range(len(filepaths_and_text)):
            # Handle both 2-column (single speaker) and 3-column (multispeaker) formats
            if len(filepaths_and_text[i]) == 3:
                # Multispeaker format: filepath|speaker_id|text
                original_text = filepaths_and_text[i][2]
                print(f"Processing: {original_text}")
                cleaned_text = english_cleaners3(original_text, language)
                filepaths_and_text[i][2] = cleaned_text
            else:
                # Single speaker format: filepath|text
                original_text = filepaths_and_text[i][1]
                print(f"Processing: {original_text}")
                cleaned_text = english_cleaners3(original_text, language)
                filepaths_and_text[i][1] = cleaned_text

        new_filelist = filelist + ".cleaned"

        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])