from transformers import get_linear_schedule_with_warmup
import torch.optim as optim


class MixedOptimizer(optim.Optimizer):

    def __init__(self, optimizers, alternating=False):
        """
        Args:
            optimizers: list of objects that are subclasses of optim.Optimizer
            alternating: whether to alternate steps with different optimizers
        """

        self.optimizers = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["method"] = type(optimizer)
                group["initial_lr"] = group.get("initial_lr", group["lr"])
            self.optimizers.append(optimizer)
        super(MixedOptimizer, self).__init__(
            (g for o in self.optimizers for g in o.param_groups), {}
        )
        self.alternating = alternating
        self.iteration = 0

    def step(self, closure=None):

        if self.alternating:
            self.optimizers[self.iteration % len(self.optimizers)].step(closure=closure)
        else:
            for optimizer in self.optimizers:
                optimizer.step(closure=closure)
        self.iteration += 1


class MixedScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, schedulers):
        """
        Args:
            optimizer: MixedOptimizer object
            schedulers: list of optim.lr_scheduler._LRScheduler objects
        """

        self.schedulers = schedulers
        super(MixedScheduler, self).__init__(optimizer)

    def step(self, epoch=None):

        for scheduler in self.schedulers:
            scheduler.step()


# class ExpGradientOptimizer(torch.optim.Optimizer):
#     def __init__(self, params, lr, window=10):
#         params = list(params)
#         self.eps = 0.01
#         self.history = []
#         self.window = window
#         for param in params:
#             if torch.abs(param.sum() - 1.0) > self.eps:
#                 param /= param.sum()  # TODO
#                 # raise(ValueError("parameters must lie on the simplex"))
#         super(ExpGradientOptimizer, self).__init__(params, {"lr": lr})

#     def step(self, losses):
#         for group in self.param_groups:
#             lr = group["lr"]
#             self.history.append(losses)
#             logit = -lr * torch.tensor(self.history[: self.window]).sum(0)

#             for p in group["params"]:
#                 p.data = torch.exp(logit.reshape(p.data.shape))
#                 p.data = normalize(p.data, -1)

#         print(p.data)

#         # # logit = -self.lr * agg_losses
#         # logit = -lr * agg_losses
#         # logit = torch.clip(logit, min=-9)
#         # for p in group["params"]:
#         #     p.data = torch.exp(logit)
#         #     p.data = normalize(p.data, -1)
