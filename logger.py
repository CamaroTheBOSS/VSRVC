import datetime

import wandb


class Logger:
    def __init__(self, interval, task_dict):
        self.interval = interval
        self.task_dict = task_dict
        self.counter = 0
        self.last_print = datetime.datetime.now()

    def log(self, task, metric_name, value, mode="train"):
        pass

    def push(self, mode="train"):
        pass

    def print(self, msg):
        if self.counter % self.interval == 0:
            print(f"[{datetime.datetime.now()}] {msg}")
        self.counter = (self.counter + 1) % self.interval


def build_metric_name(mode, task, metric):
    return f"{mode}_{task}_{metric}"


class WandbLogger(Logger):
    def __init__(self, interval, task_dict, log_grads):
        super().__init__(interval, task_dict)
        self.metric_dict = {}
        self.log_grads = log_grads
        self.train_step_metric = "Training step"
        self.test_step_metric = "Epoch"
        self.define_metrics()

    def create_metric(self, name, step_metric=None):
        if step_metric is None:
            step_metric = "Step"
        if step_metric not in self.metric_dict.keys():
            self.metric_dict[step_metric] = [{}, 0]
            wandb.define_metric(step_metric)
        self.metric_dict[step_metric][0][name] = 0
        wandb.define_metric(name, step_metric=step_metric)

    def define_metrics(self):
        if self.log_grads:
            self.create_metric(build_metric_name("grad", "vc", "norm"), step_metric=self.train_step_metric)
            self.create_metric(build_metric_name("grad", "vsr", "norm"), step_metric=self.train_step_metric)
            self.create_metric(build_metric_name("grad", "cos", "angle"), step_metric=self.train_step_metric)
        self.create_metric(build_metric_name("train", "aux", "loss"), step_metric=self.train_step_metric)
        for task in self.task_dict.keys():
            self.create_metric(build_metric_name("train", task, "loss"), step_metric=self.train_step_metric)
            for metric in self.task_dict[task]['metrics']:
                self.create_metric(build_metric_name("val", task, metric), step_metric=self.test_step_metric)

    def log(self, task, metric_name, value, mode="train"):
        if mode not in ["train", "val", "grad"]:
            raise NotImplementedError("Only train/test/grad mode is allowed")
        name = build_metric_name(mode, task, metric_name)
        step_metric = self.train_step_metric if mode in ["train", "grad"] else self.test_step_metric
        self.metric_dict[step_metric][0][name] = value

    def push(self, mode="train"):
        if mode not in ["train", "val"]:
            raise NotImplementedError("Only train/test mode is allowed")
        step_metric_name = self.train_step_metric if mode == "train" else self.test_step_metric
        data = {name: value for name, value in self.metric_dict[step_metric_name][0].items()}
        data[step_metric_name] = self.metric_dict[step_metric_name][1]
        wandb.log(data)
        self.metric_dict[step_metric_name][1] += 1

