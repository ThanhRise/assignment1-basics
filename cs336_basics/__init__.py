import importlib.metadata
from .train_bpe import run_train_bpe_backend
from .tokenizer import BPETokenizer
from .transformer import Linear, Embedding, RMSNorm, SwiGLU
__version__ = importlib.metadata.version("cs336_basics")
