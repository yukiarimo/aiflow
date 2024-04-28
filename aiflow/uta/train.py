import torch
from uta import MusicTokenizer
from model import GPTConfig, DataLoader, GPT, Trainer, GPTMusicTokenizer

def main():
    # Load the dataset from a text file
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split the dataset by commas
    notes = text.split(',')
    
    # Initialize the MusicTokenizer
    music_tokenizer = MusicTokenizer()
    
    # Tokenize each note in the dataset
    tokenized_notes = [music_tokenizer.tokenize(note) for note in notes if music_tokenizer.tokenize(note) is not None]

    # Convert the tokenized notes to a tensor
    data = torch.tensor(tokenized_notes, dtype=torch.long)
    
    # Initialize the GPTMusicTokenizer
    tokenizer = GPTMusicTokenizer(music_tokenizer)
    
    # Set up the GPT configuration
    config = GPTConfig(tokenizer.vocab_size, block_size=256)
    
    # Split the data into training and validation sets
    n = int(0.9 * len(data))
    data_loader = DataLoader({'train': data[:n], 'val': data[n:]}, config.block_size, config.batch_size, config.device)
    
    # Initialize the GPT model
    model = GPT(config).to(config.device)
    
    # Initialize the Trainer
    trainer = Trainer(model, data_loader, tokenizer, config)
    
    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model('gpt_model.pth')

if __name__ == "__main__":
    main()