from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch


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
