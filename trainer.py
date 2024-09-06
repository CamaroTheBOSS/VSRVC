import itertools
import json
import shutil
from typing import cast

import torch, os
import torch.nn as nn
import numpy as np
from compressai.entropy_models import EntropyBottleneck
from torch import Tensor

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters
import weighting as weighting_method
import archs as architecture_method
from datasets import Augmentation

from logger import Logger, WandbLogger
from scheduler import MyCosineAnnealingLR
from utils import confirm_action


def process_kwargs(kwargs=None):
    for key, value in kwargs.items():
        if isinstance(value, dict):
            return process_kwargs(value)
        elif isinstance(value, nn.Module):
            kwargs[key] = value.to_json()
        else:
            kwargs[key] = value
    return kwargs


class Trainer(nn.Module):
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, decoder_kwargs, rep_grad,
                 multi_input, optim_param, scheduler_param, logging, print_interval, lmbda, model_type, scale,
                 log_grads, dataset_type, save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()

        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.decoder_kwargs = decoder_kwargs
        self.lmbda = lmbda
        self.log_grads = log_grads
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.model_type = model_type
        self.scale = scale
        self.logger = WandbLogger(print_interval, task_dict, log_grads) if logging else Logger(print_interval, task_dict)

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)

        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)
        self.crop_size = (256, 256)
        self.augmentation = Augmentation(multi_input, scale, dataset_type)

    def augment_data(self, data, task=None):
        return self.augmentation(data, task, training_mode=self.training)

    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting = weighting_method.__dict__[weighting]
        architecture = architecture_method.__dict__[architecture]

        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device,
                                               **kwargs)
                self.init_param()

            def shared_params(self):
                for name, param in self.encoder.named_parameters():
                    if not name.endswith(".quantiles"):
                        yield param

            def get_share_params(self):
                return self.shared_params()

        self.model = MTLmodel(task_name=self.task_name,
                              encoder_class=encoder_class,
                              decoders=decoders,
                              rep_grad=self.rep_grad,
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        for decoder in self.model.decoders.values():
            try:
                decoder.share(self.model.encoder)
                print("Share function called")
            except:
                pass
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)

        if self.save_path is not None:
            if os.path.exists(self.save_path):
                if confirm_action("Path already exists, continuing will remove directory and create new one. "
                                  "Press [y/n] to continue/break\n"):
                    shutil.rmtree(self.save_path)
                else:
                    raise InterruptedError("Keyboard interruption")
            os.makedirs(self.save_path)
            model_data = {
                "task_name": self.task_name,
                "lmbda": self.lmbda,
                "encoder_class": encoder_class.__name__,
                "model_type": self.model_type,
                "scale": self.scale,
                "decoders": [
                    {
                        "task": task,
                        "module": module.__class__.__name__,
                        "kwargs": self.decoder_kwargs[task]
                    } for task, module in decoders.items()
                ],
                "weighting": weighting.__name__,
                "architecture": architecture.__name__,
                "arch_args": process_kwargs(self.kwargs['arch_args']),
                "rep_grad": self.rep_grad,
                "multi_input": self.multi_input,
                "checkpoint": os.path.abspath(os.path.join(self.save_path, "best.pt"))
            }
            with open(os.path.join(self.save_path, "model.json"), "w") as f:
                json.dump(model_data, f)

    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'adagrad': torch.optim.Adagrad,
            'rmsprop': torch.optim.RMSprop,
        }
        scheduler_dict = {
            'exp': torch.optim.lr_scheduler.ExponentialLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
            'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'mycos': MyCosineAnnealingLR,
        }
        parameters = {
            "main": {
                name
                for name, param in self.named_parameters()
                if param.requires_grad and not name.endswith(".quantiles")
            },
            "aux": {
                name
                for name, param in self.named_parameters()
                if param.requires_grad and name.endswith(".quantiles")
            },
        }
        params_dict = dict(self.named_parameters())
        inter_params = parameters["main"] & parameters["aux"]
        union_params = parameters["main"] | parameters["aux"]
        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        params_main = (params_dict[name] for name in sorted(parameters["main"]))
        self.optimizer = optim_dict[optim_param['optim']](params_main, **optim_arg)
        tmp, params_aux = itertools.tee((params_dict[name] for name in sorted(parameters["aux"])), 2)
        self.aux_optimizer = optim_dict[optim_param['optim']](params_aux, **optim_arg) if len(list(tmp)) else None

        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
            self.aux_scheduler = scheduler_dict[scheduler_param['scheduler']](self.aux_optimizer, **scheduler_arg) if self.aux_optimizer else None

        else:
            self.scheduler = None
            self.aux_scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label

    def process_preds(self, preds, task_name=None):
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
                self.logger.log(task, "loss", train_losses[tn])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
            self.logger.log(task_name, "loss", train_losses)
        return train_losses

    def _compute_aux_loss(self):
        loss = cast(Tensor, sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)))
        self.logger.log("aux", "loss", loss)
        return loss

    def _prepare_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def train(self, train_dataloaders, test_dataloaders, epochs,
              val_dataloaders=None, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs, train_gts = self._process_data(train_loader)
                    train_inputs, train_gts = self.augment_data(train_inputs)
                    train_preds = self.model(train_inputs)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.meter.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_input, train_gt = self._process_data(train_loader[task])
                        train_input, train_gt = self.augment_data(train_input, task)
                        train_pred = self.model(train_input, task)
                        train_pred = train_pred[task]
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.meter.update(train_pred, train_gt, task)

                self.optimizer.zero_grad(set_to_none=False)
                w, grads = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                if self.log_grads:
                    norm_vc = torch.norm(grads[0])
                    norm_vsr = torch.norm(grads[1])
                    cos_angle = torch.dot(grads[0], grads[1]) / (norm_vc * norm_vsr)
                    self.logger.log("vc", "norm", norm_vc, mode="grad")
                    self.logger.log("vsr", "norm", norm_vsr, mode="grad")
                    self.logger.log("cos", "angle", cos_angle, mode="grad")
                self.optimizer.step()
                if self.aux_optimizer is not None:
                    aux_loss = self._compute_aux_loss()
                    self.aux_optimizer.zero_grad(set_to_none=False)
                    aux_loss.backward()
                    self.aux_optimizer.step()
                self.logger.print(f"{batch_index + 1}/{train_batch}")
                self.logger.push()

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()

            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                    if self.aux_scheduler is not None:
                        self.aux_scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
                    if self.aux_scheduler is not None:
                        self.aux_scheduler.step()
            # if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
            print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight

    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch.
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)

        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_inputs, test_gts = self.augment_data(test_inputs)
                    test_preds = self.model(test_inputs)
                    self.meter.update(test_preds, test_gts)
                    self.logger.print(f"{batch_index + 1}/{test_batch}")
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_input, test_gt = self.augment_data(test_input, task)
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        self.meter.update(test_pred, test_gt, task)

        self.meter.record_time('end')
        self.meter.get_score()
        for task in self.task_name:
            for i, metric_name in enumerate(self.task_dict[task]["metrics"]):
                self.logger.log(task, metric_name, self.meter.results[task][i], mode="val")
        self.logger.push(mode="val")
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement
