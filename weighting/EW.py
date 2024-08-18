import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from weighting.abstract_weighting import AbsWeighting


class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """

    def __init__(self):
        super(EW, self).__init__()

    def backward(self, losses, **kwargs):
        if kwargs["log_grads"] is not None:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode="backward")
            self._reset_grad(grads.sum(0))
            return np.ones(self.task_num), grads
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return np.ones(self.task_num), None
