"""Map diffusion times to chronological state indices."""
from __future__ import annotations

from typing import List


def default_keyframe_indices(num_steps: int) -> List[int]:
    """Index T+1 states, including fully masked and fully denoised endpoints.

    State index u corresponds to diffusion time t=T-u. The paper's times
    {T, floor(T/2), floor(T/4), 0} map to
    {0, T-floor(T/2), T-floor(T/4), T}.
    """
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    return sorted({0, num_steps - num_steps // 2, num_steps - num_steps // 4, num_steps})
