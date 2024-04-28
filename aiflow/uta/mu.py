# Example usage:
import json
from uta import MusicTokenizer, Uta
from music21 import note

def process_song_from_file(input_file_path, output_file_path):
    tokenizer = MusicTokenizer()

    # load the song musicxml file
    song = Uta()
    song.load(input_file_path)
    text_representation = song.convert_musicxml_to_text()

    return process_song(text_representation, output_file_path)

def process_song(text_representation, output_file_path):
    tokenizer = MusicTokenizer()

    # take the all notes from the song and tokenize them
    song_tokenized = []
    for line in text_representation.strip().split('\n'):
        token = tokenizer.tokenize(line)
        if token is not None:
            song_tokenized.append(token)

    # Detokenize the song
    song_detokenized = []
    for token in song_tokenized:
        if token is not None:
            detokenized = tokenizer.detokenize(token)
            song_detokenized.append(detokenized)

    # put the song back together
    song = Uta()
    for token in song_detokenized:
        pitch, dur = token.split('-')
        song.addNote(pitch, dur)

    # save the song to a file in musicxml format
    song.save(output_file_path)

    return song_detokenized

# load the text_representation file and process it to a musicxml file
input_file_path = 'text_representation.txt'
output_file_path = 'musicxml'
song = process_song_from_file(input_file_path, output_file_path)