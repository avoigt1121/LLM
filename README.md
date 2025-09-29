# TinyLLM

A lightweight transformer-based language model implementation built from scratch in PyTorch. This project demonstrates the core concepts of modern language models in a simplified, educational format.

## Features

- Transformer Architecture: Multi-head self-attention mechanism with layer normalization
- Flexible Tokenization: Support for both character-level and BPE (Byte Pair Encoding) tokenization
- GPU Support: Automatic CUDA detection and utilization
- Training Monitoring: Real-time loss tracking and model statistics
- Interactive Generation: Chat-like interface for text generation
- Multiple Modes: Training, single generation, and interactive modes
- Checkpoint Management: Save and load trained models

## Project Structure

```
├── config.py          # Model and training hyperparameters
├── corpus.txt          # Training data (text corpus)
├── demo.py            # Quick demonstration script
├── generation.py      # Text generation utilities
├── main.py           # Main training and inference script
├── model.py          # TinyLLM transformer model implementation
├── tokenizer.py      # Tokenization utilities (character-level & BPE)
├── training.py       # Training loop and data loading
├── utils.py          # Helper functions and utilities
└── requirements.txt  # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd LLM
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare training data**:
   - Add your text corpus to `corpus.txt`
   - The model will train on any text file you provide

## Quick Start

### Demo Mode
Run a quick demonstration to see all components working together:

```bash
python3 demo.py
```

### Training a Model
Train TinyLLM on your corpus:

```bash
python3 main.py --mode train --data corpus.txt
```

### Generate Text
Generate text using a trained model:

```bash
python3 main.py --mode generate --checkpoint checkpoints/tinyllm_checkpoint.pt --prompt "Hello world" --length 50
```

### Interactive Mode
Start an interactive chat session:

```bash
python3 main.py --mode interactive --checkpoint checkpoints/tinyllm_checkpoint.pt
```

## Configuration

Modify `config.py` to adjust model and training parameters:

```python
# Model Architecture
EMBED_DIM = 16          # Embedding dimension
CONTEXT_SIZE = 32       # Maximum context length
NUM_HEADS = 2           # Number of attention heads
NUM_LAYERS = 1          # Number of transformer layers

# Training Parameters
BATCH_SIZE = 32         # Training batch size
LEARNING_RATE = 1e-4    # Learning rate
NUM_EPOCHS = 2000       # Number of training epochs

# Tokenization
USE_BPE = True          # Use BPE (True) or character-level (False)
```

## Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  --mode {train,generate,interactive}  Mode to run the model (default: train)
  --checkpoint PATH                    Path to model checkpoint
  --prompt TEXT                        Prompt for generation (default: "Hello world")
  --length INT                         Length of generated text (default: 20)
  --temperature FLOAT                  Temperature for sampling (default: 1.0)
  --data PATH                          Path to training data (default: corpus.txt)
```

## Model Architecture

TinyLLM implements a simplified transformer architecture:

- **Multi-Head Self-Attention**: Enables the model to focus on different parts of the input
- **Feed-Forward Networks**: Transform attention outputs
- **Layer Normalization**: Stabilizes training
- **Positional Encoding**: Provides sequence position information
- **Residual Connections**: Helps with gradient flow

## Training Process

1. **Data Loading**: Text corpus is loaded and preprocessed
2. **Tokenization**: Text is converted to tokens (character-level or BPE)
3. **Model Creation**: TinyLLM transformer is initialized
4. **Training Loop**: Model learns to predict next tokens
5. **Checkpoint Saving**: Trained model is saved for later use

## Generation Modes

### Single Generation
Generate a fixed amount of text from a prompt:
```python
generator.generate(prompt="Once upon a time", max_length=100, temperature=0.8)
```

### Interactive Generation
Continuous conversation mode where you can chat with the model:
```
> Hello, how are you?
[Model generates response]
> Tell me a story
[Model generates story]
```

## System Requirements

- **Python**: 3.7+
- **PyTorch**: 2.0.0+
- **Memory**: Minimum 4GB RAM
- **GPU**: Optional but recommended (CUDA support)

## Performance Notes

- Training time depends on corpus size and hardware
- GPU acceleration significantly improves training speed
- Model size is intentionally small for educational purposes
- Larger models require adjusting hyperparameters in `config.py`

## Educational Value

This project is designed to help understand:
- Transformer architecture fundamentals
- Self-attention mechanisms
- Language model training processes
- Tokenization strategies
- Text generation techniques

## Troubleshooting

### Common Issues

1. **"python not found"**: Use `python3` instead of `python` on macOS/Linux
2. **CUDA out of memory**: Reduce `BATCH_SIZE` in `config.py`
3. **Poor generation quality**: Train for more epochs or use a larger corpus
4. **Slow training**: Enable GPU support or reduce model size

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify your Python environment and dependencies
3. Ensure your corpus file exists and is readable
4. Try the demo script first to test the installation

## License

This project is intended for educational purposes. Feel free to use and modify as needed.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional tokenization methods
- More sophisticated generation strategies
- Training optimization techniques
- Model architecture variants

---

*Built for learning and understanding language models*
