from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def crossEntropyLoss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"], pad_token_id=-100
) -> Float[Tensor, ""]:
    batch_size = inputs.size(0)
    lse = torch.logsumexp(inputs, dim=-1)
    batch_indices = torch.arange(batch_size)
    target_logits = inputs[batch_indices, targets]
    sample_loss = -target_logits + lse
    mask = (targets != pad_token_id).float()
    sample_loss = sample_loss * mask 
    return sample_loss.mean()
