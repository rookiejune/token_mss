from typing import List
from torch.optim.lr_scheduler import LambdaLR
import torch


def calculate_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    s_true = ref
    s_artif = est - ref
    sdr = 10.0 * (
        torch.log10(torch.clip(s_true ** 2, 1e-8, torch.inf))
        - torch.log10(torch.clip(s_artif ** 2, 1e-8, torch.inf))
    ).mean()
    return sdr
