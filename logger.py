import datetime

import wandb


class Logger:
    def __init__(self, interval):
        self.interval = interval
        self.counter = 0
        self.last_print = datetime.datetime.now()

    def log(self, data):
        pass

    def push(self, data):
        pass

    def print(self, msg):
        if self.counter % self.interval == 0:
            print(f"[{datetime.datetime.now()}] {msg}")
        self.counter = (self.counter + 1) % self.interval


class WandbLogger(Logger):
    def __init__(self, interval):
        super().__init__(interval)

    def log(self, data):
        wandb.log(data, commit=False)

    def push(self, data):
        wandb.log(data, commit=True)
