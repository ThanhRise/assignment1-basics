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
            return x @ self.weight.T + self.bias
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
