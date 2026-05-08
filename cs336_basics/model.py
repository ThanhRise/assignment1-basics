from __future__ import annotations

import torch
from cs336_basics.nn import (
    Linear, Embedding, RMSNorm, SwiGLU,
    GatingFineGrainedMoE, TritonGroupedGEMMMoE, TorchGroupedMMMoE,
)
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if token_positions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            if kv_cache is not None:
                past_len = kv_cache[0].size(-2)
                positions = positions + past_len
            batch_dim = x.shape[:-2]
            token_positions = positions.expand(*batch_dim, seq_len)
            
        attn_out = self.attn(self.ln1(x), token_positions, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            attn_out, new_kv_cache = attn_out
            
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if use_cache:
            return x, new_kv_cache
        return x


class TransformerBlockMoE(torch.nn.Module):
    """Transformer block using GatingFineGrainedMoE (einsum/batched matmul)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttetion(num_heads=num_heads, d_model=d_model, theta=theta, max_seq_len=max_seq_len, **factory_kwargs)
        self.ffn = GatingFineGrainedMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if token_positions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            if kv_cache is not None:
                past_len = kv_cache[0].size(-2)
                positions = positions + past_len
            batch_dim = x.shape[:-2]
            token_positions = positions.expand(*batch_dim, seq_len)

        attn_out = self.attn(self.ln1(x), token_positions, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            attn_out, new_kv_cache = attn_out

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if use_cache:
            return x, new_kv_cache
        return x


class TransformerBlockTritonMoE(torch.nn.Module):
    """Transformer block using TritonGroupedGEMMMoE (custom Triton kernel)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttetion(num_heads=num_heads, d_model=d_model, theta=theta, max_seq_len=max_seq_len, **factory_kwargs)
        self.ffn = TritonGroupedGEMMMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if token_positions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            if kv_cache is not None:
                past_len = kv_cache[0].size(-2)
                positions = positions + past_len
            batch_dim = x.shape[:-2]
            token_positions = positions.expand(*batch_dim, seq_len)

        attn_out = self.attn(self.ln1(x), token_positions, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            attn_out, new_kv_cache = attn_out

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if use_cache:
            return x, new_kv_cache
        return x


class TransformerBlockGroupedMMMoE(torch.nn.Module):
    """Transformer block using TorchGroupedMMMoE (torch._grouped_mm, PyTorch 2.6+)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model=d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttetion(num_heads=num_heads, d_model=d_model, theta=theta, max_seq_len=max_seq_len, **factory_kwargs)
        self.ffn = TorchGroupedMMMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None, kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if token_positions is None:
            seq_len = x.size(-2)
            positions = torch.arange(seq_len, device=x.device)
            if kv_cache is not None:
                past_len = kv_cache[0].size(-2)
                positions = positions + past_len
            batch_dim = x.shape[:-2]
            token_positions = positions.expand(*batch_dim, seq_len)

        attn_out = self.attn(self.ln1(x), token_positions, kv_cache=kv_cache, use_cache=use_cache)
        if use_cache:
            attn_out, new_kv_cache = attn_out

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        if use_cache:
            return x, new_kv_cache
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

    def forward(self, input_ids: torch.Tensor, token_positions: torch.Tensor | None = None, kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if token_positions is None:
            seq_len = input_ids.size(-1)
            positions = torch.arange(seq_len, device=input_ids.device)
            if kv_cache is not None:
                past_len = kv_cache[0][0].size(-2)
                positions = positions + past_len
            batch_dim = input_ids.shape[:-1]
            token_positions = positions.expand(*batch_dim, seq_len)
            
        x = self.token_embeddings(input_ids)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            layer_out = layer(x, token_positions=token_positions, kv_cache=layer_kv_cache, use_cache=use_cache)
            if use_cache:
                x, new_layer_kv_cache = layer_out
                new_kv_cache.append(new_layer_kv_cache)
            else:
                x = layer_out
                
        x = self.ln_final(x)
        logits = self.lm_head(x)
        # Layer * (24 * N * d**2 + 4 * N**2 * d) + 2 * N * d_model * Vocab | N = batch * seq_size
        if use_cache:
            return logits, new_kv_cache
        return logits


class TransformerLM_MoE(torch.nn.Module):
    """TransformerLM using GatingFineGrainedMoE (einsum/batched matmul)."""

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = torch.nn.ModuleList([
            TransformerBlockMoE(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


class TransformerLM_TritonMoE(torch.nn.Module):
    """TransformerLM using TritonGroupedGEMMMoE (custom Triton kernel)."""

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = torch.nn.ModuleList([
            TransformerBlockTritonMoE(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)


class TransformerLM_GroupedMMMoE(torch.nn.Module):
    """TransformerLM using TorchGroupedMMMoE (torch._grouped_mm, PyTorch 2.6+)."""

    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, num_experts: int, num_experts_per_tok: int, device: torch.device | None = None, dtype: torch.dtype | None = None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.context_length = context_length
        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **factory_kwargs)
        self.layers = torch.nn.ModuleList([
            TransformerBlockGroupedMMMoE(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, **factory_kwargs)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, **factory_kwargs)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
