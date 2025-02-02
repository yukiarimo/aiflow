import json
import re
import os
import torch
from pydub import AudioSegment, silence
from transformers import pipeline
from tqdm import tqdm

class AudioDataWorker:
    def __init__(self, device="mps", model="openai/whisper-medium"):
        self.device = device
        self.model = model
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float32,
            device=self.device,
            model_kwargs={"attn_implementation": "sdpa"},
        )

    def combine_audio_files(self, input_folder, output_file, silence_duration=1000):
        audio_files = sorted(
            [f for f in os.listdir(input_folder) if f.endswith('.wav')],
            key=lambda x: int(x.split('.')[0])
        )
        combined = AudioSegment.empty()
        one_sec_silence = AudioSegment.silent(duration=silence_duration)
        for i, file in enumerate(audio_files):
            audio = AudioSegment.from_wav(os.path.join(input_folder, file))
            combined += audio
            if i < len(audio_files) - 1:
                combined += one_sec_silence
        combined.export(output_file, format="wav")
        print(f"Combined audio saved as {output_file}")

    def split_audio_into_chunks(self, input_file, output_dir, min_silence_len=400, silence_thresh=-40, min_chunk_length=4000, max_chunk_length=20000):
        os.makedirs(output_dir, exist_ok=True)
        audio = AudioSegment.from_wav(input_file)
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=500
        )
        
        def process_chunks(chunks, min_len, max_len):
            processed, current = [], AudioSegment.empty()
            for chunk in chunks:
                current += chunk
                if len(current) >= min_len:
                    if len(current) > max_len:
                        processed.append(current[:max_len])
                        current = current[max_len:]
                    else:
                        processed.append(current)
                        current = AudioSegment.empty()
            if len(current) > 0:
                processed.append(current)
            return processed

        final_chunks = process_chunks(chunks, min_chunk_length, max_chunk_length)
        for i, chunk in enumerate(final_chunks):
            chunk_filename = os.path.join(output_dir, f"chunk_{i+1}.wav")
            chunk.export(chunk_filename, format="wav")
            print(f"Exported {chunk_filename}")
        print("Splitting completed successfully!")

    def calculate_total_duration(self, directory):
        wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        total_ms = 0
        print("Calculating total duration...")
        for wav_file in tqdm(wav_files, desc="Processing files"):
            try:
                audio = AudioSegment.from_wav(os.path.join(directory, wav_file))
                total_ms += len(audio)
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
        total_seconds = total_ms / 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal duration:\nHours: {int(hours)}\nMinutes: {int(minutes)}\nSeconds: {int(seconds)}")
        print(f"Total files processed: {len(wav_files)}")

    def transcribe_audio(self, audio_file, chunk_length_s=30, batch_size=24, return_timestamps=False):
        result = self.asr_pipeline(
            audio_file,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
        )
        return result

    def transcribe_directory(self, directory_path, output_file, chunk_length_s=30, batch_size=24, return_timestamps=False):
        with open(output_file, 'w') as f:
            for filename in os.listdir(directory_path):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(directory_path, filename)
                    transcription = self.transcribe_audio(audio_path, chunk_length_s, batch_size, return_timestamps)
                    f.write(f"{audio_path}|Yuna|EN|{transcription['text'].strip()}\n")
                    print(f"Transcribed {audio_path}")

    def delete_unlisted_files(self, metadata_file, directory):
        with open(metadata_file, 'r') as file:
            lines = file.readlines()
        metadata_files = {line.strip().split('|')[0] for line in lines}
        all_files = {os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')}
        files_to_delete = all_files - metadata_files
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        print("Deletion of unlisted files completed successfully!")

    def rename_files(self, metadata_file, directory, start_number=2565):
        with open(metadata_file, 'r') as file:
            lines = file.readlines()
        metadata = [line.strip().split('|') for line in lines]
        for i, data in enumerate(metadata):
            old_path = data[0]
            new_name = f"{start_number + i}.wav"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            data[0] = new_path
            print(f"Renamed {old_path} to {new_path}")
        with open(metadata_file, 'w') as file:
            for data in metadata:
                file.write('|'.join(data) + '\n')
        print("Renaming and metadata update completed successfully!")

    def sort_transcriptions(self, input_file, output_file):
        with open(input_file, 'r') as file:
            lines = file.readlines()

        sorted_lines = sorted(lines, key=lambda x: int(x.split('/')[1].split('.')[0]))

        with open(output_file, 'w') as file:
            file.writelines(sorted_lines)

    def transliterate_file(self, input_path, output_path):
        from yuna.text.cyrillic import transliterate

        with open(input_path, 'r', encoding='utf-8') as infile:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        text = parts[3]
                        parts[3] = transliterate(text, source='ru', target='en')
                    outfile.write('|'.join(parts))

    def start_ui(self):
        transcript = input("Enter the path to the transcript file: ")

        from aiflow.AudioDataWorkerUI import main
        main(transcript)

class TextDataWorker:
    def __init__(self, device="mps"):
        self.device = device
        self.model_path = "/Users/yuki/Documents/Github/yuna-ai/lib/models/yuna/yuna-ai-v4-q6_k.gguf"

    def split_text_to_jsonl(self, input_file, output_file, mode="split_by_block", block="<SPLITTEXT>", chunk_size=16384):
        from llama_cpp import Llama
        llm = Llama(model_path=self.model_path)

        # Open the input file and read the content
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()

        if mode == "split_by_block":
            # Split the content by the specified block into a list of chunks
            chunks = re.split(block, content)

            # Open the output file in write mode
            with open(output_file, 'w', encoding='utf-8') as file:
                for chunk in chunks:
                    note = chunk.strip()
                    json_object = {"text": note}
                    file.write(json.dumps(json_object) + '\n')

        elif mode == "split_by_token_chunks":
            # Tokenize the content
            tokenized_text = llm.tokenize(content.encode('utf-8'), special=True, add_bos=False)

            # Open the output file in write mode
            with open(output_file, 'w', encoding='utf-8') as file:
                for i in range(0, len(tokenized_text), chunk_size):
                    chunk_tokens = tokenized_text[i:i+chunk_size]
                    chunk_text = llm.detokenize(chunk_tokens, special=True).decode('utf-8')
                    json_object = {"text": chunk_text}
                    file.write(json.dumps(json_object) + '\n')
    
    def count_tokens(self, input_file):
        from llama_cpp import Llama
        llm = Llama(model_path=self.model_path)
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        tokens = llm.tokenize(content.encode('utf-8'))
        print(f"Total tokens: {len(tokens)}")

    def convert_jsonl_to_text(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines:
                data = json.loads(line)
                file.write(data['text'] + '\n')

    def convert_docx_to_text(self, input_file, output_file):
        from docx import Document

        doc = Document(input_file)
        processed_text = []

        for para in doc.paragraphs:
            for run in para.runs:
                if run.italic:
                    processed_text.append(f"**{run.text}**")
                else:
                    processed_text.append(run.text)
            processed_text.append('\n')  # Add a newline after each paragraph

        with open(output_file, 'w') as f:
            f.write(''.join(processed_text))