import torch


class ScheduledOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, warm_up_scheduler = None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warm_up_scheduler = warm_up_scheduler

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, scaler) -> None:
        if scaler:
            scaler.step(self.optimizer)
        else:
            self.optimizer.step()

    def sched_step(self) -> None:
        if self.warm_up_scheduler:
            self.warm_up_scheduler.step()
            if self.warm_up_scheduler.finished():
                self.warm_up_scheduler = None
        else:
            self.scheduler.step()

class WarmUpScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, epochs) -> None:
        self.optimizer = optimizer
        self.init_lr = self.optimizer.param_groups[0]["lr"]
        self.lr_interval = self.init_lr / epochs
        self.optimizer.param_groups[0]["lr"] = self.lr_interval

    def step(self):
        self.optimizer.param_groups[0]["lr"] += self.lr_interval

    def finished(self) -> bool:
        return self.optimizer.param_groups[0]["lr"] > self.init_lr - self.lr_interval


def create_adamw_polylr_optim(
    parameters, lr, betas, weight_decay, epochs, factor
) -> ScheduledOptimizer:
    optim = torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, epochs, factor)
    return ScheduledOptimizer(optim, scheduler)


def create_adamw_polylr_warmup_optim(
    parameters, lr, betas, weight_decay, epochs, factor, warm_up
):
    optim = torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optim, epochs - warm_up, factor)
    warm_up_scheduler = WarmUpScheduler(optim, warm_up)
    return ScheduledOptimizer(optim, scheduler, warm_up_scheduler=warm_up_scheduler)