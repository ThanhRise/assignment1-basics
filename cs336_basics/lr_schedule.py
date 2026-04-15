from __future__ import annotations

import math


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
