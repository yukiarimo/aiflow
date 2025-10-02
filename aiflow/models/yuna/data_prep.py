import json
import torch
from model.yuna import Yuna
from model.processor import Processor

# Load the model and processor (needed for tokenizer)
model_name = "Yuna-2B"
processor = Processor(repo_id=model_name, vision_config=None)  # No vision needed for text processing
tokenizer = processor.tokenizer

def process_text_file(input_file, output_prefix, chunk_sizes):
    # Read the text file
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the entire text
    input_ids = tokenizer.encode(text, add_special_tokens=False).ids

    for chunk_size in chunk_sizes:
        chunks = []
        for i in range(0, len(input_ids), chunk_size):
            chunk_tokens = input_ids[i:i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=False)
            chunks.append({"text": "<|endoftext|>" + chunk_text})

        # Save as JSONL
        output_file = f"{output_prefix}_{chunk_size}.jsonl"
        with open(output_file, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"Created {len(chunks)} chunks of up to {chunk_size} tokens for {input_file}, saved to {output_file}")

if __name__ == "__main__":
    chunk_sizes = [2048, 4096, 8192, 16384]

    process_text_file("yuna-ai-dataset/dialog.txt", "output_dialog", chunk_sizes)

    # Concatenate saved files into combined single file
    for chunk_size in chunk_sizes:
        combined_file = f"output_combined_{chunk_size}.jsonl"
        with open(combined_file, "w", encoding="utf-8") as outfile:
            for prefix in ["output_dialog"]:
                input_file = f"{prefix}_{chunk_size}.jsonl"
                with open(input_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
        print(f"Created combined file {combined_file} for chunk size {chunk_size}")

    print("Data preparation completed.")