from collections.abc import Iterable

import torch


class AdamWOptimizer:
    """
    Used for updating model parameters using the AdamW optimization algorithm.
    Args:
    parameters: Model parameters to optimize.
    learning_rate: Learning rate used to scale parameter updates.
    weight_decay: Rate at which parameters are decayed toward zero.
    beta_1: Decay rate for the moving average of gradients. 
    beta_2: Decay rate for the moving average of squared gradients.
    Both betas are used to scale parameter updates based on gradient history. beta_1 is for momentum, beta_2 is for magnitude.
    """
    def __init__(self, parameters: Iterable[torch.Tensor], learning_rate: float, weight_decay: float, beta_1: float, beta_2: float):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = [torch.zeros_like(p) for p in self.parameters]  # First moment
        self.v = [torch.zeros_like(p) for p in self.parameters]  # Second moment
        self.t = 0  # Time step


    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if (gradient := param.grad) is None:
                continue

            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gradient
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (gradient * gradient)
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

            with torch.no_grad():
                param *= (1 - self.learning_rate * self.weight_decay)
                param -= self.learning_rate * (m_hat / (torch.sqrt(v_hat) + 1e-8))