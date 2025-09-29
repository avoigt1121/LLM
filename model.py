"""Neural network models for TinyLLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # expand
            nn.ReLU(),  # nonlinearity
            nn.Linear(embed_dim * 4, embed_dim)  # contract back
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        T = x.size(1)
        # Create causal mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # Multi-head attention with residual connection
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)
        
        # Feed-forward network with residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x


class TinyLLM(nn.Module):
    """Small transformer-based language model."""
    
    def __init__(self, vocab_size: int, embed_dim: int, context_size: int, num_heads: int):
        super().__init__()
        self.context_size = context_size
        
        # Embedding layers
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_size, embed_dim)
        
        # Transformer block
        self.transformer = TransformerBlock(embed_dim, num_heads)
        
        # Output layer with weight tying
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # weight tying for efficiency
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.token_embed(idx)  # (B, T, E)
        
        # Positional embeddings
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))  # (T, E)
        
        # Combine embeddings
        x = tok_emb + pos_emb  # (B, T, E)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def estimate_memory_usage(self) -> str:
        """Estimate memory usage of the model."""
        num_params = self.get_num_params()
        # Assuming float32 (4 bytes per parameter)
        memory_mb = (num_params * 4) / (1024 * 1024)
        return f"{memory_mb:.2f} MB"
