from __future__ import annotations

import math
import torch
from einops import einsum


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
        if self.bias is not None:
            return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out") + self.bias
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
        self.weight = torch.nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **factory_kwargs))

        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            self.weight.register_hook(self._zero_padding_gradients)

        self.reset_parameter()
                
    def _zero_padding_gradients(self, grad: torch.Tensor) -> torch.Tensor:
        grad_out = grad.clone()
        grad_out[self.padding_idx].fill_(0.0)
        return grad_out

    def reset_parameter(self) -> None:
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

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
        self.w1 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, self.d_model, **factory_kwargs)
        self.w3 = Linear(self.d_model, self.d_ff, **factory_kwargs)
        self.reset_parameter()

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.sigmoid(x)

    def reset_parameter(self) -> None:
        self.w1.reset_parameter()
        self.w2.reset_parameter()
        self.w3.reset_parameter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  self.w2(self.silu(self.w1(x)) * (self.w3(x)))

class GatingFineGrainedMoE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        factory_kwargs = {"device": device, "dtype": dtype}

        # Router
        self.gate = Linear(d_model, num_experts, **factory_kwargs)

        # All expert weights stacked: [num_experts, out_dim, in_dim]
        # SwiGLU has 3 weight matrices: w1, w3 (up-projections), w2 (down-projection)
        self.w1 = torch.nn.Parameter(
            torch.empty(num_experts, d_ff, d_model, **factory_kwargs)
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(num_experts, d_model, d_ff, **factory_kwargs)
        )
        self.w3 = torch.nn.Parameter(
            torch.empty(num_experts, d_ff, d_model, **factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for w in (self.w1, self.w2, self.w3):
            for i in range(self.num_experts):
                # Match the same truncated normal init as your Linear class
                fan_in, fan_out = w.shape[2], w.shape[1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                torch.nn.init.trunc_normal_(
                    w.data[i], mean=0, std=std, a=-3 * std, b=3 * std
                )

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)         # [T, d_model]
        T = x_flat.shape[0]

        # ── 1. Gating ──────────────────────────────────────────
        gate_logits = self.gate(x_flat)            # [T, E]
        top_k_logits, top_k_indices = torch.topk(
            gate_logits, self.num_experts_per_tok, dim=-1
        )                                          # [T, K], [T, K]
        top_k_weights = top_k_logits.softmax(dim=-1)  # [T, K]

        # ── 2. Flatten token-expert pairs ──────────────────────
        # Each token has K expert assignments → T*K total pairs
        flat_indices = top_k_indices.view(-1)      # [T*K]  expert id per pair
        flat_weights = top_k_weights.view(-1)      # [T*K]  weight per pair
        # Which token each pair belongs to
        token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(-1, self.num_experts_per_tok).reshape(-1)  # [T*K]
        flat_x = x_flat[token_ids]                 # [T*K, d_model]

        # ── 3. Gather expert weights for each pair ─────────────
        # Instead of looping, index into stacked weights
        # w1[flat_indices] → [T*K, d_ff, d_model]
        w1 = self.w1[flat_indices]                 # [T*K, d_ff, d_model]
        w2 = self.w2[flat_indices]                 # [T*K, d_model, d_ff]
        w3 = self.w3[flat_indices]                 # [T*K, d_ff, d_model]

        # ── 4. Batched SwiGLU (einsum) ─────────────────────────
        h1 = einsum(w1, flat_x, "p d_ff d_model, p d_model -> p d_ff")
        h3 = einsum(w3, flat_x, "p d_ff d_model, p d_model -> p d_ff")
        hidden = self.silu(h1) * h3                # [T*K, d_ff]
        out = einsum(w2, hidden, "p d_model d_ff, p d_ff -> p d_model")

        # ── 5. Weighted scatter-add ────────────────────────────
        weighted_out = einsum(flat_weights, out, "p, p d_model -> p d_model")
        final_output = torch.zeros_like(x_flat)    # [T, d_model]
        final_output.scatter_add_(
            0,
            token_ids.unsqueeze(-1).expand_as(weighted_out),
            weighted_out,
        )

        return final_output.view(original_shape)


# ════════════════════════════════════════════════════════════════════
# Triton Grouped-GEMM MoE
# ════════════════════════════════════════════════════════════════════

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_gemm_kernel(
        # Pointers to the contiguous sorted-token tensor, stacked weight tensor, and output tensor
        X_ptr, W_ptr, Y_ptr,
        # Pointer to the expert-offset array (cumulative token counts) [E]
        offs_ptr,
        # Dimensions
        N: tl.constexpr,   # output dim (columns of W)
        K: tl.constexpr,   # input  dim (columns of X / rows of W)
        stride_x_row,      # stride between rows of X
        stride_w_expert,    # stride between experts in W  (= N * K)
        stride_w_row,       # stride between rows of one expert's weight (= K for row-major)
        stride_y_row,       # stride between rows of Y
        num_experts,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Persistent grouped-GEMM kernel.
        Computes Y[offs[e-1]:offs[e], :] = X[offs[e-1]:offs[e], :] @ W[e].T
        for each expert e, where W[e] has shape [N, K] (output_dim, input_dim).
        """
        pid = tl.program_id(0)

        # ── Build a tile schedule across all experts ───────────
        # We iterate over experts and find which tiles belong to this program
        tile_id = 0
        for e in range(num_experts):
            # Expert e's token range: [start_row, end_row)
            start_row = tl.load(offs_ptr + e - 1) if e > 0 else 0
            end_row = tl.load(offs_ptr + e)
            num_rows = end_row - start_row

            num_m_tiles = tl.cdiv(num_rows, BLOCK_M)
            num_n_tiles = tl.cdiv(N, BLOCK_N)
            num_tiles = num_m_tiles * num_n_tiles

            # Check if this program's tile falls within this expert
            while pid >= tile_id and pid < tile_id + num_tiles:
                # Compute tile coordinates within this expert
                local_tile = pid - tile_id
                tile_m = local_tile // num_n_tiles
                tile_n = local_tile % num_n_tiles

                # Row / col offsets
                offs_m = start_row + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                # Mask for out-of-bounds rows
                mask_m = offs_m < end_row
                mask_n = offs_n < N

                # Accumulate in fp32
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                # W[e] base pointer: W_ptr + e * stride_w_expert
                w_base = W_ptr + e * stride_w_expert

                for k_start in range(0, K, BLOCK_K):
                    k_offs = k_start + offs_k
                    mask_k = k_offs < K

                    # Load X tile: [BLOCK_M, BLOCK_K]
                    x = tl.load(
                        X_ptr + offs_m[:, None] * stride_x_row + k_offs[None, :],
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    # Load W[e] tile: [BLOCK_N, BLOCK_K] (W is stored as [N, K])
                    w = tl.load(
                        w_base + offs_n[:, None] * stride_w_row + k_offs[None, :],
                        mask=mask_n[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    # X @ W.T  →  [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
                    acc += tl.dot(x, tl.trans(w))

                # Store result
                y = acc.to(X_ptr.dtype.element_ty)
                tl.store(
                    Y_ptr + offs_m[:, None] * stride_y_row + offs_n[None, :],
                    y,
                    mask=mask_m[:, None] & mask_n[None, :],
                )

                # Advance to next tile assigned to this program (stride by total programs)
                pid += tl.num_programs(0)

            tile_id += num_tiles

    def _triton_grouped_gemm(
        x_sorted: torch.Tensor,
        w_stacked: torch.Tensor,
        expert_offsets: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        """
        x_sorted:       [total_tokens, K]   – tokens sorted by expert
        w_stacked:      [E, N, K]           – stacked expert weights
        expert_offsets:  [E]                 – cumulative token counts (int32)
        Returns:        [total_tokens, N]
        """
        total_tokens, K = x_sorted.shape
        E, N, _ = w_stacked.shape
        y = torch.empty((total_tokens, N), device=x_sorted.device, dtype=x_sorted.dtype)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        # Total number of tiles across all experts
        num_tiles = 0
        offsets_cpu = expert_offsets.cpu().tolist()
        prev = 0
        for off in offsets_cpu:
            num_rows = off - prev
            num_tiles += math.ceil(num_rows / BLOCK_M) * math.ceil(N / BLOCK_N)
            prev = off

        grid = (num_tiles,)

        _grouped_gemm_kernel[grid](
            x_sorted, w_stacked, y,
            expert_offsets,
            N=N, K=K,
            stride_x_row=x_sorted.stride(0),
            stride_w_expert=w_stacked.stride(0),
            stride_w_row=w_stacked.stride(1),
            stride_y_row=y.stride(0),
            num_experts=num_experts,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return y

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


class TritonGroupedGEMMMoE(torch.nn.Module):
    """
    MoE using a custom Triton grouped-GEMM kernel.

    Instead of duplicating weights per token-expert pair (as bmm/einsum does),
    this sorts tokens by expert and launches a single Triton kernel that
    processes all expert groups in one shot — no weight duplication, no Python loop.

    Requires: `triton` package and a CUDA GPU.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not HAS_TRITON:
            raise ImportError("Triton is required for TritonGroupedGEMMMoE")
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        factory_kwargs = {"device": device, "dtype": dtype}

        self.gate = Linear(d_model, num_experts, **factory_kwargs)
        # Weights stored as [E, out_dim, in_dim] — NOT duplicated per token
        self.w1 = torch.nn.Parameter(torch.empty(num_experts, d_ff, d_model, **factory_kwargs))
        self.w2 = torch.nn.Parameter(torch.empty(num_experts, d_model, d_ff, **factory_kwargs))
        self.w3 = torch.nn.Parameter(torch.empty(num_experts, d_ff, d_model, **factory_kwargs))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in (self.w1, self.w2, self.w3):
            for i in range(self.num_experts):
                fan_in, fan_out = w.shape[2], w.shape[1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                torch.nn.init.trunc_normal_(w.data[i], mean=0, std=std, a=-3 * std, b=3 * std)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        T = x_flat.shape[0]

        # ── 1. Gating ──────────────────────────────────────────
        gate_logits = self.gate(x_flat)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=-1)
        top_k_weights = top_k_logits.softmax(dim=-1)

        # ── 2. Flatten and sort by expert ──────────────────────
        flat_expert_ids = top_k_indices.view(-1)                    # [T*K]
        flat_weights = top_k_weights.view(-1)                       # [T*K]
        token_ids = (
            torch.arange(T, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts_per_tok)
            .reshape(-1)
        )                                                           # [T*K]

        # Sort by expert for contiguous memory access
        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_token_ids = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_x = x_flat[sorted_token_ids]                        # [T*K, d_model]

        # Compute cumulative offsets per expert
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int64)
        expert_counts.scatter_add_(0, sorted_expert_ids.long(), torch.ones_like(sorted_expert_ids, dtype=torch.int64))
        expert_offsets = expert_counts.cumsum(0).to(torch.int32)    # [E]

        # ── 3. Grouped GEMM SwiGLU ─────────────────────────────
        h1 = _triton_grouped_gemm(sorted_x, self.w1, expert_offsets, self.num_experts)  # [T*K, d_ff]
        h3 = _triton_grouped_gemm(sorted_x, self.w3, expert_offsets, self.num_experts)  # [T*K, d_ff]
        hidden = self.silu(h1) * h3
        out = _triton_grouped_gemm(hidden, self.w2, expert_offsets, self.num_experts)    # [T*K, d_model]

        # ── 4. Weighted scatter-add (unsort) ───────────────────
        weighted_out = sorted_weights.unsqueeze(-1) * out
        final_output = torch.zeros_like(x_flat)
        final_output.scatter_add_(
            0,
            sorted_token_ids.unsqueeze(-1).expand_as(weighted_out),
            weighted_out,
        )
        return final_output.view(original_shape)


# ════════════════════════════════════════════════════════════════════
# torch._grouped_mm MoE  (PyTorch 2.6+)
# ════════════════════════════════════════════════════════════════════

class TorchGroupedMMMoE(torch.nn.Module):
    """
    MoE using PyTorch's built-in torch._grouped_mm.

    Same sort-by-expert strategy as the Triton version, but delegates the
    grouped matmul to PyTorch's native fused kernel (SM >= 80, bfloat16).

    API: torch._grouped_mm(mat_a, mat_b, offs=offs)
      - mat_a: [total_tokens, K]     (2D, contiguous)
      - mat_b: [E, N, K]             (3D stacked weights)
      - offs:  [E]                   (int32 cumulative token counts)
      → returns: [total_tokens, N]
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_experts_per_tok: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        factory_kwargs = {"device": device, "dtype": dtype}

        self.gate = Linear(d_model, num_experts, **factory_kwargs)
        # Weights: [E, out_dim, in_dim]  — kept stacked, never duplicated
        self.w1 = torch.nn.Parameter(torch.empty(num_experts, d_ff, d_model, **factory_kwargs))
        self.w2 = torch.nn.Parameter(torch.empty(num_experts, d_model, d_ff, **factory_kwargs))
        self.w3 = torch.nn.Parameter(torch.empty(num_experts, d_ff, d_model, **factory_kwargs))
        self._reset_parameters()

    def _reset_parameters(self):
        for w in (self.w1, self.w2, self.w3):
            for i in range(self.num_experts):
                fan_in, fan_out = w.shape[2], w.shape[1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                torch.nn.init.trunc_normal_(w.data[i], mean=0, std=std, a=-3 * std, b=3 * std)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    @staticmethod
    def _grouped_mm(
        x: torch.Tensor,
        w: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Wrapper around torch._grouped_mm.
        x:    [total_tokens, in_dim]
        w:    [E, out_dim, in_dim]
        offs: [E] int32 cumulative offsets
        → [total_tokens, out_dim]
        """
        # torch._grouped_mm computes: x[offs[i-1]:offs[i]] @ w[i].T
        return torch._grouped_mm(x, w, offs=offs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        T = x_flat.shape[0]

        # ── 1. Gating ──────────────────────────────────────────
        gate_logits = self.gate(x_flat)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.num_experts_per_tok, dim=-1)
        top_k_weights = top_k_logits.softmax(dim=-1)

        # ── 2. Flatten and sort by expert ──────────────────────
        flat_expert_ids = top_k_indices.view(-1)
        flat_weights = top_k_weights.view(-1)
        token_ids = (
            torch.arange(T, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.num_experts_per_tok)
            .reshape(-1)
        )

        sorted_order = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_token_ids = token_ids[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_x = x_flat[sorted_token_ids].contiguous()  # must be contiguous for _grouped_mm

        # Cumulative offsets per expert (int32 required)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.int64)
        expert_counts.scatter_add_(0, sorted_expert_ids.long(), torch.ones_like(sorted_expert_ids, dtype=torch.int64))
        expert_offsets = expert_counts.cumsum(0).to(torch.int32)

        # ── 3. Grouped-MM SwiGLU ───────────────────────────────
        h1 = self._grouped_mm(sorted_x, self.w1, expert_offsets)   # [T*K, d_ff]
        h3 = self._grouped_mm(sorted_x, self.w3, expert_offsets)   # [T*K, d_ff]
        hidden = (self.silu(h1) * h3).contiguous()
        out = self._grouped_mm(hidden, self.w2, expert_offsets)     # [T*K, d_model]

        # ── 4. Weighted scatter-add (unsort) ───────────────────
        weighted_out = sorted_weights.unsqueeze(-1) * out
        final_output = torch.zeros_like(x_flat)
        final_output.scatter_add_(
            0,
            sorted_token_ids.unsqueeze(-1).expand_as(weighted_out),
            weighted_out,
        )
        return final_output.view(original_shape)