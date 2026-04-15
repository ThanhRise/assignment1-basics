from __future__ import annotations

import torch

from cs336_basics.nn import Linear, Embedding, RMSNorm, SwiGLU
from cs336_basics.attention import MultiHeadSelfAttetion


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=self.d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model=self.d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttetion(num_heads=self.num_heads, d_model=self.d_model, theta=theta, max_seq_len=max_seq_len, **factory_kwargs)
        self.ffn = SwiGLU(d_model=self.d_model, d_ff=self.d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device= x.device)
            batch_dim = x.shape[:-2]
            token_positions = positions.expand(*batch_dim, seq_len)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, max_seq_len=self.context_length, theta=rope_theta, **factory_kwargs) for _ in range(self.num_layers)])
        self.ln_final = RMSNorm(d_model=self.d_model)
        self.lm_head = Linear(in_features=self.d_model, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if token_positions is None:
            seq_len = input_ids.size(-1)
            positions = torch.arange(seq_len, device=input_ids.device)
            batch_dim = input_ids.shape[:-1]
            token_positions = positions.expand(*batch_dim, seq_len)
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        # Layer * (24 * N * d**2 + 4 * N**2 * d) + 2 * N * d_model * Vocab | N = batch * seq_size
        return logits
