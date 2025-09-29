"""Text generation utilities for TinyLLM."""

import torch
import torch.nn.functional as F
from typing import List
from model import TinyLLM
from tokenizer import TokenizerManager
from config import DEVICE


class TextGenerator:
    """Text generator for TinyLLM."""
    
    def __init__(self, model: TinyLLM, tokenizer_manager: TokenizerManager):
        self.model = model
        self.tokenizer_manager = tokenizer_manager
        self.model.eval()  # Set to evaluation mode
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
        context_size: int = None
    ) -> str:
        """Generate text given a prompt."""
        # Encode the prompt
        context = torch.tensor(
            [self.tokenizer_manager.encode(prompt)], 
            dtype=torch.long
        ).to(DEVICE)
        
        if context_size is None:
            context_size = self.model.context_size
        
        # Generate tokens
        with torch.no_grad():
            for _ in range(max_length):
                # Truncate context to fit model's context size
                input_context = context[:, -context_size:]
                
                # Get model predictions
                logits = self.model(input_context)
                
                # Apply temperature scaling
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Convert to probabilities and sample
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                
                # Append to context
                context = torch.cat([context, next_id], dim=1)
        
        # Decode and return
        generated_ids = context[0].tolist()
        return self.tokenizer_manager.decode_tokens(generated_ids)
    
    def generate_multiple(
        self, 
        prompt: str, 
        num_samples: int = 3,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
        context_size: int = None
    ) -> List[str]:
        """Generate multiple text samples from the same prompt."""
        samples = []
        for _ in range(num_samples):
            sample = self.generate(
                prompt, 
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                context_size=context_size
            )
            samples.append(sample)
        return samples
    
    def interactive_generation(self, context_size: int = None) -> None:
        """Interactive text generation session."""
        print("Interactive Text Generation")
        print("Type 'quit' to exit, 'help' for options")
        print("-" * 40)
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                elif prompt.lower() == 'help':
                    self._show_help()
                    continue
                elif not prompt:
                    print("Please enter a prompt or 'quit' to exit.")
                    continue
                
                # Parse generation parameters (simple format)
                params = self._parse_params(prompt)
                
                generated = self.generate(
                    params.get('prompt', prompt),
                    max_length=params.get('length', 20),
                    temperature=params.get('temp', 1.0),
                    top_k=params.get('top_k'),
                    context_size=context_size
                )
                
                print(f"\nGenerated: {generated}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self) -> None:
        """Show help for interactive generation."""
        help_text = """
Available commands:
- quit: Exit the session
- help: Show this help message

Generation parameters (append to prompt):
- --length N: Set max generation length (default: 20)
- --temp N: Set temperature (default: 1.0)
- --top_k N: Set top-k filtering

Example: Hello world --length 30 --temp 0.8
        """
        print(help_text)
    
    def _parse_params(self, text: str) -> dict:
        """Parse generation parameters from text."""
        parts = text.split('--')
        prompt = parts[0].strip()
        params = {'prompt': prompt}
        
        for part in parts[1:]:
            if part.startswith('length'):
                try:
                    params['length'] = int(part.split()[1])
                except (IndexError, ValueError):
                    pass
            elif part.startswith('temp'):
                try:
                    params['temp'] = float(part.split()[1])
                except (IndexError, ValueError):
                    pass
            elif part.startswith('top_k'):
                try:
                    params['top_k'] = int(part.split()[1])
                except (IndexError, ValueError):
                    pass
        
        return params
