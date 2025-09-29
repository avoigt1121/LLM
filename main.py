"""Main script to run TinyLLM training and generation."""

import argparse
import time
from pathlib import Path

# Local imports
from config import *
from tokenizer import TokenizerManager, load_and_preprocess_data
from model import TinyLLM
from training import DataLoader, Trainer
from generation import TextGenerator
from utils import (
    print_model_summary, 
    get_device_info, 
    validate_config, 
    create_project_structure,
    format_time
)


def main():
    """Main function to train and run TinyLLM."""
    parser = argparse.ArgumentParser(description='TinyLLM - A small transformer language model')
    parser.add_argument('--mode', choices=['train', 'generate', 'interactive'], 
                       default='train', help='Mode to run the model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='Hello world', 
                       help='Prompt for generation')
    parser.add_argument('--length', type=int, default=20, 
                       help='Length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for generation sampling')
    parser.add_argument('--data', type=str, default=DATA_FILE,
                       help='Path to training data file')
    
    args = parser.parse_args()
    
    # Print system information
    print("TinyLLM - A Small Transformer Language Model")
    print("=" * 50)
    device_info = get_device_info()
    print(f"Device: {device_info['current_device']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory']})")
    print()
    
    # Create project structure
    create_project_structure()
    
    # Validate configuration
    config_dict = {
        'EMBED_DIM': EMBED_DIM,
        'CONTEXT_SIZE': CONTEXT_SIZE,
        'NUM_HEADS': NUM_HEADS,
        'NUM_LAYERS': NUM_LAYERS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_EPOCHS': NUM_EPOCHS
    }
    validate_config(config_dict)
    
    # Load and preprocess data
    print(f"Loading data from {args.data}...")
    try:
        data = load_and_preprocess_data(args.data)
        print(f"Data loaded successfully. Length: {len(data)} characters")
    except FileNotFoundError:
        print(f"Error: Data file '{args.data}' not found.")
        return
    
    # Setup tokenizer
    print(f"Setting up tokenizer (BPE: {USE_BPE})...")
    tokenizer_manager = TokenizerManager(use_bpe=USE_BPE)
    tokenizer_manager.setup_tokenizer(data)
    
    print(f"Vocabulary size: {tokenizer_manager.vocab_size}")
    
    # Prepare training data
    train_ids = tokenizer_manager.encode(data)
    print(f"Training tokens: {len(train_ids)}")
    
    # Create model
    print("\nCreating model...")
    model = TinyLLM(
        vocab_size=tokenizer_manager.vocab_size,
        embed_dim=EMBED_DIM,
        context_size=CONTEXT_SIZE,
        num_heads=NUM_HEADS
    ).to(DEVICE)
    
    print_model_summary(model)
    
    if args.mode == 'train':
        # Training mode
        print(f"\nStarting training...")
        start_time = time.time()
        
        # Create data loader and trainer
        data_loader = DataLoader(train_ids)
        trainer = Trainer(
            model=model,
            data_loader=data_loader,
            learning_rate=LEARNING_RATE,
            vocab_size=tokenizer_manager.vocab_size
        )
        
        # Train the model
        trainer.train(
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            context_size=CONTEXT_SIZE,
            print_interval=PRINT_INTERVAL
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(training_time)}")
        
        # Save checkpoint
        checkpoint_path = "checkpoints/tinyllm_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Print training stats
        stats = trainer.get_training_stats()
        print("\nTraining Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")
    
    # Generation modes
    if args.mode in ['generate', 'interactive'] or args.checkpoint:
        generator = TextGenerator(model, tokenizer_manager)
        
        if args.mode == 'generate':
            # Single generation
            print(f"\nGenerating text with prompt: '{args.prompt}'")
            generated = generator.generate(
                prompt=args.prompt,
                max_length=args.length,
                temperature=args.temperature,
                context_size=CONTEXT_SIZE
            )
            print(f"Generated: {generated}")
        
        elif args.mode == 'interactive':
            # Interactive generation
            generator.interactive_generation(context_size=CONTEXT_SIZE)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
