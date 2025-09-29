"""Tokenization utilities for TinyLLM."""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from typing import List, Callable, Tuple
from config import SPECIAL_TOKENS


class TokenizerManager:
    """Manages both BPE and character-level tokenization."""
    
    def __init__(self, use_bpe: bool = True):
        self.use_bpe = use_bpe
        self.tokenizer = None
        self.vocab_size = 0
        self.encode_fn = None
        self.decode_fn = None
        self._setup_complete = False
    
    def setup_tokenizer(self, data: str) -> None:
        """Setup tokenizer based on the provided data."""
        if self.use_bpe:
            self._setup_bpe_tokenizer(data)
        else:
            self._setup_char_tokenizer(data)
        self._setup_complete = True
    
    def _setup_bpe_tokenizer(self, data: str) -> None:
        """Setup BPE tokenizer."""
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=SPECIAL_TOKENS)
        self.tokenizer.train_from_iterator([data], trainer)
        
        self.encode_fn = lambda s: self.tokenizer.encode(s).ids
        self.decode_fn = lambda l: self.tokenizer.decode(l)
        self.vocab_size = self.tokenizer.get_vocab_size()
    
    def _setup_char_tokenizer(self, data: str) -> None:
        """Setup character-level tokenizer."""
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        self.encode_fn = lambda s: [self.stoi[c] for c in s]
        self.decode_fn = lambda l: ''.join([self.itos[i] for i in l])
        self.vocab_size = len(self.stoi)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self._setup_complete:
            raise RuntimeError("Tokenizer not setup. Call setup_tokenizer() first.")
        return self.encode_fn(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self._setup_complete:
            raise RuntimeError("Tokenizer not setup. Call setup_tokenizer() first.")
        return self.decode_fn(token_ids)
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Robust decode for generation."""
        if self.use_bpe:
            return self.tokenizer.decode(list(token_ids))
        else:
            return ''.join([self.itos.get(i, '?') for i in token_ids])


def load_and_preprocess_data(file_path: str) -> str:
    """Load and preprocess text data from file."""
    with open(file_path, "r") as f:
        data = f.read().replace("\n", " ")
    return data
