import torch
from uta import MusicTokenizer
from model import GPTConfig, GPTMusicTokenizer, GPT, Trainer

def main():
    # Initialize the MusicTokenizer
    music_tokenizer = MusicTokenizer()
    
    # Initialize the GPTMusicTokenizer
    tokenizer = GPTMusicTokenizer(music_tokenizer)
    
    # Set up the GPT configuration
    config = GPTConfig(tokenizer.vocab_size, block_size=256)
    
    # Initialize the GPT model
    model = GPT(config).to(config.device)
    
    # Initialize the Trainer
    trainer = Trainer(model, None, tokenizer, config)
    
    # Load the model
    trainer.load_model('checkpoint_3800.pth')

    # Define an initial sequence of tokens (for example, the encoded representation of a melody)
    initial_notes = ['C4-quarter', 'E4-eighth', 'G4-eighth', 'A4-16th', 'F4-16th', 'E4-quarter', 'D4-eighth', 'B3-eighth', 'C4-16th', 'A3-16th']  # Replace with actual notes you want to start with
    initial_tokens = tokenizer.encode(initial_notes)

    # Generate new music tokens starting from the initial sequence
    generated_tokens = trainer.generate(initial_tokens, max_new_tokens=500)
    
    # Decode the generated tokens into music notes
    generated_notes = tokenizer.decode(generated_tokens[0].tolist())
    print(generated_notes)

if __name__ == "__main__":
    main()