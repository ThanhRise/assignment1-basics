from __future__ import annotations

import os
from collections.abc import Iterable, Callable
from typing import IO, Any, BinaryIO, Optional

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
    
def crossEntropyLoss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    batch_size = inputs.size(0)
    lse = torch.logsumexp(inputs, dim=-1)
    batch_indices = torch.arange(batch_size)
    target_logits = inputs[batch_indices, targets]
    sample_loss = -target_logits + lse
    return sample_loss.mean()


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: 
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
        return loss

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0, eps= 1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 or 1: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(f"CustomAdamW does not support sparse gradient.")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m_t = state["m"]
                v_t = state["v"]
                state["step"]+=1
                step = state["step"]

                # m_t = beta1 * state["m"] + (1 - beta1) * grad
                m_t.mul_(beta1).add_(grad, alpha =1.0 - beta1)
                # v_t = beta2 * state["v"] + (1 - beta2) * grad**2
                v_t.mul_(beta2).addcmul_(grad, grad, value = 1.0 - beta2)
                
                bias_connection1 = 1.0 - beta1 ** step
                bias_connection2 = 1.0 - beta2 ** step
                
                alpha_t = lr * math.sqrt(bias_connection2) / bias_connection1

                p.addcdiv_(m_t, v_t.sqrt().add_(eps), value = -alpha_t) 
                

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
        return loss

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it <= warmup_iters:
        if warmup_iters ==0: return max_learning_rate
        return it / warmup_iters * max_learning_rate
    if warmup_iters <= it and it <= cosine_cycle_iters:
        return min_learning_rate + (1 + math.cos((it - warmup_iters)/(cosine_cycle_iters-warmup_iters)* math.pi))*(max_learning_rate - min_learning_rate) / 2.0
    return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    eps = 1e-6
    
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0: return 0.0

    total_norm = 0.0
    for p in params_with_grad:
        param_norm = p.grad.detach().norm(2)
        total_norm += param_norm**2
    total_norm = total_norm**0.5
    if total_norm >= max_l2_norm:
        clip_factor = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_factor)
    return 

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample. 
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data = torch.from_numpy(dataset).long().to(device)
    batch = data.unfold(dimension=0, size=context_length, step=1)
    sampled_idx = torch.randperm(len(batch)-1)[:batch_size]
    sampled = batch[sampled_idx]
    lable = batch[sampled_idx + 1]
    return sampled, lable


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iteration = checkpoint['iteration']
    return iteration