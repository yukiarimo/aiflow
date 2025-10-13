import json
import random
import re
from pathlib import Path

def parse_chat_dialog(txt_content):
    """Parse the chat dialog from txt format."""
    conversations = []
    lines = txt_content.strip().split('\n')

    current_chat = []
    for line in lines:
        # Extract role and content using regex
        yuki_match = re.match(r'<yuki>(.*?)</yuki>', line)
        yuna_match = re.match(r'<yuna>(.*?)</yuna>', line)

        if yuki_match:
            current_chat.append({
                "role": "yuki",
                "content": yuki_match.group(1)
            })
        elif yuna_match:
            current_chat.append({
                "role": "yuna",
                "content": yuna_match.group(1)
            })

    return current_chat

def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def create_mixed_dataset(chat_txt_path, images_jsonl_path, audio_jsonl_path, output_path, num_samples=20):
    """Create mixed dataset with different modalities."""

    # Load data
    with open(chat_txt_path, 'r', encoding='utf-8') as f:
        chat_content = f.read()

    chat_dialog = parse_chat_dialog(chat_content)
    images_data = load_jsonl(images_jsonl_path)
    audio_data = load_jsonl(audio_jsonl_path)

    dataset = []

    # Add pure chat conversations (split into pairs)
    for i in range(0, len(chat_dialog) - 1, 2):
        dataset.append({
            "chat": [chat_dialog[i], chat_dialog[i + 1]],
            "image_paths": [],
            "audio_paths": []
        })

    # Add image-based conversations
    image_prompts = [
        "Please tell me what can you see here, Yuna?",
        "What do you see in this image, Yuna?",
        "Can you describe this for me?",
        "What's in this picture?"
    ]

    for img_data in random.sample(images_data, min(len(images_data), num_samples)):
        prompt = random.choice(image_prompts)
        dataset.append({
            "chat": [
                {
                    "role": "yuki",
                    "content": f"{prompt} <|vision_start|><|image_pad|><|vision_end|>"
                },
                {
                    "role": "yuna",
                    "content": img_data["description"]
                }
            ],
            "image_paths": [img_data["path"]],
            "audio_paths": []
        })

    # Add audio-based conversations
    audio_prompts = [
        "Please tell me what do you hear, Yuna?",
        "What do you hear in this audio?",
        "Can you describe this sound for me?",
        "What's playing here?"
    ]

    for audio_item in random.sample(audio_data, min(len(audio_data), num_samples)):
        prompt = random.choice(audio_prompts)
        dataset.append({
            "chat": [
                {
                    "role": "yuki",
                    "content": f"{prompt} <|vision_start|><|quad_start|><|vision_end|>"
                },
                {
                    "role": "yuna",
                    "content": audio_item["description"]
                }
            ],
            "image_paths": [],
            "audio_paths": [audio_item["path"]]
        })

    # Add some multi-modal examples (image + audio)
    num_multimodal = min(5, len(images_data), len(audio_data))
    for i in range(num_multimodal):
        img_data = random.choice(images_data)
        audio_item = random.choice(audio_data)

        dataset.append({
            "chat": [
                {
                    "role": "yuki",
                    "content": "Look at this <|vision_start|><|image_pad|><|vision_end|> and listen to this <|vision_start|><|quad_start|><|vision_end|>"
                },
                {
                    "role": "yuna",
                    "content": f"{img_data['description']} And {audio_item['description'].lower()}"
                }
            ],
            "image_paths": [img_data["path"]],
            "audio_paths": [audio_item["path"]]
        })

    # Shuffle the dataset
    random.shuffle(dataset)

    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"   - Pure chat: {len([d for d in dataset if not d['image_paths'] and not d['audio_paths']])}")
    print(f"   - With images: {len([d for d in dataset if d['image_paths']])}")
    print(f"   - With audio: {len([d for d in dataset if d['audio_paths']])}")
    print(f"   - Multi-modal: {len([d for d in dataset if d['image_paths'] and d['audio_paths']])}")
    print(f"   Saved to: {output_path}")

if __name__ == "__main__":
    # Update these paths to match your files
    chat_txt_path = "/Users/yuki/Documents/Github/yuna-ai/lib/models/yuna/yuna-ai-dataset/dialog-humanized-WITH-TRASH.txt"
    images_jsonl_path = "/Users/yuki/Downloads/Sounds/image.jsonl"
    audio_jsonl_path = "/Users/yuki/Downloads/Sounds/sounds.jsonl"
    output_path = "mixed_dataset.jsonl"

    create_mixed_dataset(
        chat_txt_path,
        images_jsonl_path,
        audio_jsonl_path,
        output_path,
        num_samples=20000  # Adjust how many image/audio samples to include
    )