import importlib.metadata
from .train_bpe import run_train_bpe_backend
from .tokenizer import BPETokenizer
from .nn import Linear, Embedding, RMSNorm, SwiGLU
from .attention import softMax, scaled_dot_product_attention, RoPE, MultiHeadSelfAttetion
from .model import TransformerBlock, TransformerLM
from .loss import crossEntropyLoss
from .optimizer import SGDOptimizer, AdamWOptimizer
from .lr_schedule import lr_cosine_schedule
from .utils import gradient_clipping, get_batch, save_checkpoint, load_checkpoint
from .data import ShardedDataset, process_to_shards
__version__ = importlib.metadata.version("cs336_basics")
