from LibMTL.loss import L1Loss

import wandb
from datasets import Vimeo90k

from example.trainer import Trainer
from models.vsrvc import VSRVCEncoder, VSRDecoder
from utils import *
from aspp import DeepLabHead
from create_dataset import NYUv2

from LibMTL.model import resnet_dilated
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args


def parse_args(parser):
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=8, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    parser.add_argument('--enable_wandb', action="store_true", default=False, help='whether to enable wandb')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    # prepare dataloaders
    nyuv2_train_set = Vimeo90k("../../Datasets/VIMEO90k", 2, sliding_window_size=1)
    nyuv2_test_set = Vimeo90k("../../Datasets/VIMEO90k", 2, test_mode=True, sliding_window_size=1)

    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=params.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=params.test_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    # define tasks
    task_dict = {'vc': {'metrics': ['mIoU', 'pixAcc'],
                        'metrics_fn': SegMetric(),
                        'loss_fn': L1Loss(),
                        'weight': [1, 1]},
                 'vsr': {'metrics': ['abs_err', 'rel_err'],
                         'metrics_fn': DepthMetric(),
                         'loss_fn': L1Loss(),
                         'weight': [0, 0]}}

    # define encoder and decoders
    def encoder_class():
        return VSRVCEncoder()
        # return resnet_dilated('resnet50')

    num_out_channels = {'segmentation': 13, 'depth': 1, 'normal': 3}
    decoders = nn.ModuleDict({task: VSRDecoder(64, 64) for task in list(task_dict.keys())})
    if params.enable_wandb:
        run_name = f"NYU TEST"
        wandb.init(project="VSRVC", name=run_name)
    NYUmodel = Trainer(task_dict=task_dict,
                       weighting=params.weighting,
                       architecture=params.arch,
                       encoder_class=encoder_class,
                       decoders=decoders,
                       rep_grad=params.rep_grad,
                       multi_input=params.multi_input,
                       optim_param=optim_param,
                       scheduler_param=scheduler_param,
                       save_path=params.save_path,
                       load_path=params.load_path,
                       logging=params.enable_wandb,
                       print_interval=2,
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
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)
