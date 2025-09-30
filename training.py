"""Training utilities for TinyLLM."""

import torch
import torch.nn.functional as F
from typing import List, Tuple
from model import TinyLLM
from config import DEVICE

class DataLoader:
    """Simple data loader for language modeling."""
    
    def __init__(self, train_ids: List[int]):
        # Pre-convert to tensor for efficiency
        self.train_tensor = torch.tensor(train_ids, dtype=torch.long)
        self.data_length = len(train_ids)
    
    def get_batch(self, batch_size: int, context_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of training data."""
        ix = torch.randint(self.data_length - context_size, (batch_size,))
        
        # Efficient batch creation using pre-converted tensor
        x = torch.stack([self.train_tensor[i:i+context_size] for i in ix])
        y = torch.stack([self.train_tensor[i+1:i+context_size+1] for i in ix])
        
        return x.to(DEVICE), y.to(DEVICE)


class Trainer:
    """Trainer class for TinyLLM."""
    
    def __init__(
        self, 
        model: TinyLLM, 
        data_loader: DataLoader,
        learning_rate: float = 1e-4,
        vocab_size: int = None
    ):
        self.model = model
        self.data_loader = data_loader
        self.vocab_size = vocab_size or model.vocab_size
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = None  # Will be initialized in train() method
        
        # Training metrics
        self.losses = []
        self.current_epoch = 0
    
    def train_step(self, batch_size: int, context_size: int) -> float:
        """Perform one training step."""
        # Get batch
        xb, yb = self.data_loader.get_batch(batch_size, context_size)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.model(xb)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size), 
            yb.view(-1)
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def train(
        self, 
        num_epochs: int, 
        batch_size: int, 
        context_size: int,
        print_interval: int = 50
    ) -> None:
        """Train the model for specified number of epochs."""
        # Initialize scheduler with actual number of epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Estimated memory usage: {self.model.estimate_memory_usage()}")
        
        for epoch in range(num_epochs):
            loss = self.train_step(batch_size, context_size)
            
            if len(self.losses) < 1000:  # Keep only last 1000 losses
                self.losses.append(loss)
            else:
                self.losses = self.losses[1:] + [loss]  # Rolling window
            self.current_epoch = epoch
            if epoch % 100 == 0:
                torch.cuda.empty_cache() 
            if epoch % print_interval == 0:
                print(f"Epoch {epoch}, loss {loss:.4f}")
        
        print("Training completed!")
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.current_epoch,
            'losses': self.losses
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.losses = checkpoint['losses']
        print(f"Checkpoint loaded from {filepath}")
    
    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if not self.losses:
            return {"message": "No training performed yet"}
        
        return {
            "epochs_trained": self.current_epoch + 1,
            "final_loss": self.losses[-1],
            "best_loss": min(self.losses),
            "avg_loss": sum(self.losses) / len(self.losses)
        }
 