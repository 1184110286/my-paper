from __future__ import annotations

import torch

from malsnif.utils.cuda_memory import cuda_memory_stats, reset_cuda_peak


def test_cuda_memory_stats_cpu_empty():
    reset_cuda_peak(torch.device('cpu'))
    assert cuda_memory_stats(torch.device('cpu')) == {}
