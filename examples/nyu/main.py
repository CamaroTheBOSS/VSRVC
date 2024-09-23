import os

from utils_nyu import *
from aspp import DeepLabHead
from create_dataset import NYUv2

from LibMTL import Trainer
from LibMTL.model import resnet_dilated

from torch.utils.data import DataLoader

from LibMTL.utils import set_device, set_random_seed
from config import MyLibMTL_args, prepare_args
from trainer import Trainer
import wandb


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--enable_wandb', action='store_true', default=False, help='wandb')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--log_grads', action='store_true', default=False, help='dataset path')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    kwargs["arch_args"]["encoder_kwargs"] = {}

    # prepare dataloaders
    nyuv2_train_set = NYUv2(root=params.dataset_path, mode='train', augmentation=params.aug)
    nyuv2_test_set = NYUv2(root=params.dataset_path, mode='test', augmentation=False)

    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # define tasks
    task_dict = {'segmentation': {'metrics': ['mIoU', 'pixAcc'],
                                  'metrics_fn': SegMetric(),
                                  'loss_fn': SegLoss(),
                                  'weight': [1, 1]},
                 'depth': {'metrics': ['abs_err', 'rel_err'],
                           'metrics_fn': DepthMetric(),
                           'loss_fn': DepthLoss(),
                           'weight': [0, 0]}}

    # define encoder and decoders
    def encoder_class():
        return resnet_dilated('resnet50')

    num_out_channels = {'segmentation': 13, 'depth': 1}
    decoders = nn.ModuleDict({task: DeepLabHead(2048,
                                                num_out_channels[task]) for task in list(task_dict.keys())})
    params.save_path = os.path.join(params.save_path, "Segmentation+Depth DBMTL")

    if params.enable_wandb:
        wandb.init(project="VSRVC", name="Segmentation+Depth DBMTL")
    NYUmodel = Trainer(task_dict=task_dict,
                       weighting=params.weighting,
                       architecture=params.arch,
                       encoder_class=encoder_class,
                       decoders=decoders,
                       decoder_kwargs={task: {} for task in task_dict.keys()},
                       rep_grad=params.rep_grad,
                       multi_input=params.multi_input,
                       optim_param=optim_param,
                       logging=params.enable_wandb,
                       scale=69,
                       print_interval=2,
                       lmbda=0,
                       dataset_type="NYU",
                       log_grads=False,
                       model_type="segmantation+depth",
                       scheduler_param=scheduler_param,
                       save_path=params.save_path,
                       load_path=params.load_path,
                       **kwargs)
    if params.mode == 'train':
        NYUmodel.train(nyuv2_train_loader, nyuv2_test_loader, params.epochs)
    elif params.mode == 'test':
        NYUmodel.test(nyuv2_test_loader)
    else:
        raise ValueError
    if params.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    params = parse_args(MyLibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
