import torch

class NoamOpt:
    def __init__(self, optimizer, model_size, factor, warmup, last_epoch=-1):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.last_epoch = last_epoch
        self._lr = 0

    def step(self):
        self.last_epoch += 1
        lr = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
        self.optimizer.step()

    def learning_rate(self):
        return self.factor * (self.model_size ** (-0.5) * 
               min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def lr(self):
        return self._lr
