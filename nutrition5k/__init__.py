import torch
from torch import Tensor


def n5kloss(output: Tensor, target: Tensor, use_macronutrients=True) -> Tensor:
    batch_size = output.shape[0]
    calories_diff = torch.abs(output[:, 0] - target[:, 0])
    total_mass_diff = torch.abs(output[:, 1] - target[:, 1])
    total_loss = calories_diff + total_mass_diff
    if use_macronutrients:
        macro_diff = torch.abs(output[:, 2:] - target[:, 2:])
        total_loss += torch.mean(macro_diff, dim=1)
    return torch.sum(total_loss) / batch_size
