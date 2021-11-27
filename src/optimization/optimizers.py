import torch.optim


class NormalizedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1.5):
        default = {'lr': lr}
        super().__init__(params, default)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                p.data -= (lr / torch.mean(torch.abs(p.grad))) * p.grad
