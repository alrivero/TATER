import math
from torch.optim.lr_scheduler import _LRScheduler

class CustomWarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, min_lr, max_lr, warmup_steps, total_steps, last_epoch=-1):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.annealing_steps = total_steps - warmup_steps
        super(CustomWarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        # Calculate the current step
        current_step = self.last_epoch + 1
        self.last_epoch = current_step  # Update to the new step

        # Compute learning rate based on warmup and cosine annealing phases
        if current_step <= self.warmup_steps:
            # Linear warmup phase
            lr = self.min_lr + (self.max_lr - self.min_lr) * (current_step / self.warmup_steps)
        else:
            # Cosine annealing phase
            annealing_step = current_step - self.warmup_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * annealing_step / self.annealing_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cos_decay

        # Set the computed learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        # Return the most recent learning rate
        return [param_group['lr'] for param_group in self.optimizer.param_groups]