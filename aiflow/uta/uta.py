import music21
import json

class MusicConverter:
    def __init__(self):
        self.note_to_number = {}
        self.number_to_note = {}
        self.create_note_mappings()

    def create_note_mappings(self):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octaves = range(11)  # Assuming a range of 11 octaves

        number = 0
        for octave in octaves:
            for note in notes:
                note_name = f"{note}{octave}"
                self.note_to_number[note_name] = number
                self.number_to_note[number] = note_name
                number += 1

        # Add rest as a special token
        self.note_to_number['rest'] = number
        self.number_to_note[number] = 'rest'

    def musicxml_to_numbers(self, musicxml_file):
        # Load the MusicXML file
        score = music21.converter.parse(musicxml_file)

        # Extract notes and rests from the score
        notes_and_rests = []
        for element in score.flat.notesAndRests:
            if isinstance(element, music21.note.Note):
                note_name = element.nameWithOctave
                if note_name in self.note_to_number:
                    note_number = self.note_to_number[note_name]
                    duration = float(element.duration.quarterLength)  # Convert duration to float
                    notes_and_rests.append((note_number, duration))
                else:
                    print(f"Warning: Skipping note {note_name} as it is not found in the note mappings.")
            elif isinstance(element, music21.note.Rest):
                rest_number = self.note_to_number['rest']
                duration = float(element.duration.quarterLength)  # Convert duration to float
                notes_and_rests.append((rest_number, duration))

        return notes_and_rests

    def numbers_to_musicxml(self, numbers, output_file):
        # Create a new stream
        stream = music21.stream.Stream()

        # Iterate over the numbers and durations
        for number, duration in numbers:
            if number == self.note_to_number['rest']:
                # Create a rest with the specified duration
                rest = music21.note.Rest(quarterLength=duration)
                stream.append(rest)
            else:
                # Create a note with the specified pitch and duration
                note_name = self.number_to_note[number]
                note = music21.note.Note(note_name, quarterLength=duration)
                stream.append(note)

        # Write the stream to a MusicXML file
        stream.write('musicxml', fp=output_file)

    def save_numbers_to_file(self, numbers, file_path):
        with open(file_path, 'w') as file:
            json.dump(numbers, file)

    def load_numbers_from_file(self, file_path):
        with open(file_path, 'r') as file:
            numbers = json.load(file)
        return numbers

# Example usage
converter = MusicConverter()

# Convert MusicXML to numbers
musicxml_file = 'Base.musicxml'
numbers = converter.musicxml_to_numbers(musicxml_file)

# Save the numbers to a file
converter.save_numbers_to_file(numbers, 'numbers.json')

# Load the numbers from a file
loaded_numbers = converter.load_numbers_from_file('numbers.json')

# Convert numbers back to MusicXML
output_file = 'output.musicxml'
converter.numbers_to_musicxml(loaded_numbers, output_file)