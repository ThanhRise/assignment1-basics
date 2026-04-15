from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


def crossEntropyLoss(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, " ..."], pad_token_id=-100
) -> Float[Tensor, ""]:
    # Flatten to handle (batch_size, sequence_length) for LMs
    inputs = inputs.view(-1, inputs.size(-1))
    targets = targets.view(-1)
    
    batch_size = inputs.size(0)
    lse = torch.logsumexp(inputs, dim=-1)
    
    # clamp targets to 0 to prevent IndexError from pad_token_id (-100)
    safe_targets = targets.clamp(min=0)
    batch_indices = torch.arange(batch_size, device=inputs.device)
    target_logits = inputs[batch_indices, safe_targets]
    
    sample_loss = -target_logits + lse
    mask = (targets != pad_token_id).float()
    sample_loss = sample_loss * mask 
    
    # Sum over masked loss, and divide by the number of valid tokens
    return sample_loss.sum() / mask.sum().clamp(min=1.0)
