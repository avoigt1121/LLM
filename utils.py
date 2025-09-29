"""Utility functions for TinyLLM."""

import torch
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any
from pathlib import Path


def plot_training_losses(losses: List[float], save_path: str = None) -> None:
    """Plot training losses over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def save_config(config_dict: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file."""
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, torch.device):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    # Convert device string back to torch.device
    if 'DEVICE' in config:
        config['DEVICE'] = torch.device(config['DEVICE'])
    
    return config


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a summary of the model architecture."""
    print("Model Summary:")
    print("=" * 50)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params
            print(f"{name}: {module.__class__.__name__} - {num_params:,} parameters")
    
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    
    # Estimate memory usage
    memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
    print(f"Estimated memory usage: {memory_mb:.2f} MB")


def create_project_structure(base_path: str = ".") -> None:
    """Create recommended project structure."""
    base_path = Path(base_path)
    
    directories = [
        "checkpoints",
        "logs", 
        "outputs",
        "data"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(exist_ok=True)
    
    print("Project structure created:")
    for directory in directories:
        print(f"  📁 {directory}/")


def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def validate_config(config_dict: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_keys = [
        'EMBED_DIM', 'CONTEXT_SIZE', 'NUM_HEADS', 'NUM_LAYERS',
        'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS'
    ]
    
    for key in required_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate values
    if config_dict['EMBED_DIM'] % config_dict['NUM_HEADS'] != 0:
        raise ValueError("EMBED_DIM must be divisible by NUM_HEADS")
    
    if config_dict['CONTEXT_SIZE'] <= 0:
        raise ValueError("CONTEXT_SIZE must be positive")
    
    if config_dict['BATCH_SIZE'] <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    
    print("Configuration validation passed ✓")


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    }
    
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name()
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    
    return info
