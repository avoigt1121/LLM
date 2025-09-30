"""Demo script showing how to use the refactored TinyLLM."""

import torch
from tokenizer import TokenizerManager, load_and_preprocess_data
from model import TinyLLM
from training import DataLoader, Trainer
from generation import TextGenerator
from config import *

def quick_demo():
    """Quick demonstration of the TinyLLM functionality."""
    print("TinyLLM Quick Demo")
    print("=" * 30)
    
    # 1. Load and tokenize data
    print("1. Loading and tokenizing data...")
    data = load_and_preprocess_data("corpus.txt")
    
    tokenizer_manager = TokenizerManager(use_bpe=USE_BPE)
    tokenizer_manager.setup_tokenizer(data)
    
    train_ids = tokenizer_manager.encode(data)
    print(f"   Vocabulary size: {tokenizer_manager.vocab_size}")
    print(f"   Training tokens: {len(train_ids)}")
    
    # 2. Create model
    print("\n2. Creating model...")
    model = TinyLLM(
        vocab_size=tokenizer_manager.vocab_size,
        embed_dim=EMBED_DIM,
        context_size=CONTEXT_SIZE,
        num_heads=NUM_HEADS
    ).to(DEVICE)
    print(f"   Parameters: {model.get_num_params():,}")
    
    # 3. Quick training (just a few epochs for demo)
    data_loader = DataLoader(train_ids)
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        learning_rate=LEARNING_RATE,
        vocab_size=tokenizer_manager.vocab_size
    )
    
    trainer.train(
        num_epochs=NUM_EPOCHS,  # Just for demo
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        print_interval=25
    )
    
    # 4. Generate text
    print("\n4. Generating text...")
    generator = TextGenerator(model, tokenizer_manager)
    
    prompts = ["Hello world", "The quick brown", "Once upon a time"]
    
    for prompt in prompts:
        generated = generator.generate(
            prompt=prompt,
            max_length=15,
            temperature=0.8,
            context_size=CONTEXT_SIZE
        )
        print(f"   '{prompt}' -> '{generated}'")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    quick_demo()
