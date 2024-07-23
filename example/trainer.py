from LibMTL import Trainer
import torch.nn.functional as F


class NYUtrainer(Trainer):
    def __init__(self, task_dict, weighting, architecture, encoder_class,
                 decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
        super(NYUtrainer, self).__init__(task_dict=task_dict,
                                         weighting=weighting,
                                         architecture=architecture,
                                         encoder_class=encoder_class,
                                         decoders=decoders,
                                         rep_grad=rep_grad,
                                         multi_input=multi_input,
                                         optim_param=optim_param,
                                         scheduler_param=scheduler_param,
                                         **kwargs)

    def process_preds(self, preds, task_name=None):
        img_size = (288, 384)
        for task in self.task_name:
            preds[task] = F.interpolate(preds[task], img_size, mode='bilinear', align_corners=True)
        return preds