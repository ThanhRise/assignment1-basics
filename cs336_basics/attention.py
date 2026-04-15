from __future__ import annotations

import math
import torch
from jaxtyping import Bool, Float
from torch import Tensor
from einops import rearrange, einsum, repeat

from cs336_basics.nn import Linear


def softMax(x: torch.Tensor, dim: int = -1, Temperature: float = 1):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    stable_x = (x - max_val) / Temperature
    exp_x = torch.exp(stable_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    prob = exp_x / sum_exp
    return prob

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    d_k = Q.size(-1)
    # scores = (Q @ K.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attention_weight = softMax(scores, dim=-1)

    return  einsum(attention_weight, V, " ... queries values, ... values d_v -> ... queries d_v")


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (self.theta ** ((torch.arange(0, self.d_k, 2, dtype=torch.float32))/ self.d_k))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions].to(x.dtype)
        sin = self.sin_cache[token_positions].to(x.dtype)

        x_rotated = (x * cos) + (self.rotate_half(x) * sin)
        return x_rotated
    
    
class MultiHeadSelfAttetion(torch.nn.Module):
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self, num_heads: int, d_model: int, theta:float, max_seq_len:int = 4096, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        factory_kwargs = {"device" : device, "dtype": dtype}
        assert d_model % num_heads ==0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, **factory_kwargs)
        self.k_proj = Linear(self.d_model, self.d_model, **factory_kwargs)
        self.v_proj = Linear(self.d_model, self.d_model, **factory_kwargs)
        self.output_proj = Linear(self.d_model, self.d_model, **factory_kwargs)
                   
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

        self.reset_parameter()

    def reset_parameter(self) -> None:
        self.q_proj.reset_parameter()
        self.k_proj.reset_parameter()
        self.v_proj.reset_parameter()
        self.output_proj.reset_parameter()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #  in_features: Float[Tensor, " ... sequence_length d_in"]
        seq_len = x.size(-2)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim = self.head_dim)
        k = rearrange(k, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim = self.head_dim)
        v = rearrange(v, "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim", num_heads=self.num_heads, head_dim = self.head_dim)

        # Important token_position must be have same batch-like(batch, num_heads) with q and k
        token_positions_batch = repeat(token_positions, "... seq_len -> ... num_heads seq_len", num_heads = self.num_heads)
        q = self.rope(q, token_positions_batch)
        k = self.rope(k, token_positions_batch)

        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()
        attention = scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)

        attention = rearrange(attention, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")
        return  self.output_proj(attention)
