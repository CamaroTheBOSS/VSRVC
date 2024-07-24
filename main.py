from LibMTL.model import resnet_dilated
from torch import nn
from torch.utils.data import DataLoader

from LibMTL.loss import L1Loss
from LibMTL.utils import set_device, set_random_seed
from config import MyLibMTL_args, prepare_args
from datasets import Vimeo90k
from trainer import Trainer
from metrics import CompressionTaskMetrics, RateDistortionLoss, QualityMetrics, VSRLoss
from models.vsrvc import VCDecoder, VSRDecoder, VSRVCEncoder
import wandb


def parse_args(parser):
    parser.add_argument('--scale', default=2, type=int, help='super-resolution scaling')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training')
    parser.add_argument('--epochs', default=30, type=int, help='training epochs')
    parser.add_argument('--lmbda', default=512, type=int, help='distortion/compression ratio')
    parser.add_argument('--vimeo_path', type=str, help='path to vimeo90k dataset')
    parser.add_argument('--num_workers', default=0, type=int, help='num workers in dataloaders')
    parser.add_argument('--enable_wandb', action='store_true', default=False, help='whether to enable wandb')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)

    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=1)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=1)
    train_dataloader = DataLoader(
        train_set,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # define tasks
    task_dict = {
        'vc': {'metrics': ['psnr', 'ssim', 'bpp'],
               'metrics_fn': CompressionTaskMetrics(),
               'loss_fn': RateDistortionLoss(params.lmbda),
               'weight': [1, 1, 0]},
        'vsr': {'metrics': ['psnr', 'ssim'],
                'metrics_fn': QualityMetrics(),
                'loss_fn': VSRLoss(params.lmbda),
                'weight': [1, 1]},
    }

    def encoder_class():
        return VSRVCEncoder(in_channels=3, mid_channels=64, out_channels=64)

    decoders = nn.ModuleDict({
        'vc': VCDecoder(64, 64),
        'vsr': VSRDecoder(64, 64)
    })

    if params.enable_wandb:
        run_name = f"VSRVC MTL with bpp lambda={params.lmbda}"
        wandb.init(project="VSRVC", name=run_name)
    my_trainer = Trainer(task_dict=task_dict,
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
                         print_interval=5,
                         **kwargs)
    if params.mode == 'train':
        my_trainer.train(train_dataloader, test_dataloader, params.epochs)
    elif params.mode == 'test':
        my_trainer.test(test_dataloader)
    else:
        raise ValueError
    if params.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    params = parse_args(MyLibMTL_args)
    set_device(params.gpu_id)
    set_random_seed(params.seed)
    main(params)
