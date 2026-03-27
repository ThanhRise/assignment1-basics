from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange, einsum, repeat
import math

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


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device" : device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameter()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias:
            return x @ self.weight.T + self.bias
        # return x @ self.weight.T
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def reset_parameter(self) -> None:
        std = math.sqrt(2.0/(self.out_features+self.in_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        if self.bias is not None:
            fan_in , _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt((fan_in if fan_in > 0 else 0))
            torch.nn.init.uniform_(self.bias, -bound, bound)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **factory_kwargs))

        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            self.embedding.register_hook(self._zero_padding_gradients)

        self.reset_parameter()
                
    def _zero_padding_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        grad_out = grad.clone()
        grad_out[self.padding_idx].fill_(0.0)
        return grad_out

    def reset_parameter(self) -> None:
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embedding[self.padding_idx].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device : torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(torch.ones(self.d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x **2, dim= -1, keepdim=True) + self.eps)
        result =  (x/rms) * self.weight
        return result.to(dtype=dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = torch.nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.w2 = torch.nn.Parameter(torch.empty((self.d_model, self.d_ff), **factory_kwargs))
        self.w3 = torch.nn.Parameter(torch.empty((self.d_ff, self.d_model), **factory_kwargs))
        self.reset_parameter()

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.sigmoid(x)

    def reset_parameter(self) -> None:
        fan_in_w1 , fan_out_w1 = torch.nn.init._calculate_fan_in_and_fan_out(self.w1)
        fan_in_w2 , fan_out_w2 = torch.nn.init._calculate_fan_in_and_fan_out(self.w2)
        fan_in_w3 , fan_out_w3 = torch.nn.init._calculate_fan_in_and_fan_out(self.w3)
        std_w1 = math.sqrt(2 / (fan_in_w1 + fan_out_w1))
        std_w2 = math.sqrt(2 / (fan_in_w2 + fan_out_w2))
        std_w3 = math.sqrt(2 / (fan_in_w3 + fan_out_w3))
        torch.nn.init.trunc_normal_(self.w1, mean=0, std=std_w1, a=-3*std_w1, b=3*std_w1)
        torch.nn.init.trunc_normal_(self.w2, mean=0, std=std_w2, a=-3*std_w2, b=3*std_w2)
        torch.nn.init.trunc_normal_(self.w3, mean=0, std=std_w3, a=-3*std_w3, b=3*std_w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @  self.w2.T
    
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

        self.q_proj = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.k_proj = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.v_proj = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
        self.o_proj = torch.nn.Parameter(torch.empty((self.d_model, self.d_model), **factory_kwargs))
                   
        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

        self.reset_parameter()

    def reset_parameter(self) -> None:
        fan_in_w_q , fan_out_w_q = torch.nn.init._calculate_fan_in_and_fan_out(self.q_proj)
        fan_in_w_k , fan_out_w_k = torch.nn.init._calculate_fan_in_and_fan_out(self.k_proj)
        fan_in_w_v , fan_out_w_v = torch.nn.init._calculate_fan_in_and_fan_out(self.v_proj)
        fan_in_w_o , fan_out_w_o = torch.nn.init._calculate_fan_in_and_fan_out(self.o_proj)
        std_w_q = math.sqrt(2 / (fan_in_w_q + fan_out_w_q))
        std_w_k = math.sqrt(2 / (fan_in_w_k + fan_out_w_k))
        std_w_v = math.sqrt(2 / (fan_in_w_v + fan_out_w_v))
        std_w_o = math.sqrt(2 / (fan_in_w_o + fan_out_w_o))
        torch.nn.init.trunc_normal_(self.q_proj, mean=0, std=std_w_q, a=-3*std_w_q, b=3*std_w_q)
        torch.nn.init.trunc_normal_(self.k_proj, mean=0, std=std_w_k, a=-3*std_w_k, b=3*std_w_k)
        torch.nn.init.trunc_normal_(self.v_proj, mean=0, std=std_w_v, a=-3*std_w_v, b=3*std_w_v)
        torch.nn.init.trunc_normal_(self.o_proj, mean=0, std=std_w_o, a=-3*std_w_o, b=3*std_w_o)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #  in_features: Float[Tensor, " ... sequence_length d_in"]
        seq_len = x.size(-2)

        q = einsum(x, self.q_proj, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        k = einsum(x, self.k_proj, "... seq_len d_in, d_k d_in -> ... seq_len d_k")
        v = einsum(x, self.v_proj, "... seq_len d_in, d_k d_in -> ... seq_len d_k")

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
        
        return  einsum(attention, self.o_proj, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
    
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

    def forward(self, x: torch.Tensor, token_postions: torch.Tensor | None = None) -> torch.Tensor:
        if token_postions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device= x.device)
            batch_dim = x.shape[:-2]
            token_postions = positions.expand(*batch_dim, seq_len)
        x = x + self.attn(self.ln1(x), token_postions)
        x = x + self.ffn(self.ln2(x))
        return x