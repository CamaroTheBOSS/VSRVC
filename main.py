import os
from typing import Dict

from torch.utils.data import DataLoader

from LibMTL.utils import set_device, set_random_seed
from config import MyLibMTL_args, prepare_args
from trainer import Trainer
from metrics import CompressionTaskMetrics, RateDistortionLoss, QualityMetrics, VSRLoss, DummyMetrics, DummyLoss
import wandb
from training_configs import vsrvc, vsrvc_motion_residual, vsrvc_shallow_encoder, vsrvc_basic


def parse_args(parser):
    parser.add_argument('--scale', default=2, type=int, help='super-resolution scaling')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training')
    parser.add_argument('--epochs', default=30, type=int, help='training epochs')
    parser.add_argument('--lmbda', default=512, type=int, help='distortion/compression ratio')
    parser.add_argument('--vimeo_path', type=str, help='path to vimeo90k dataset, if provided vimeo will be used for training')
    parser.add_argument('--reds_path', type=str, help='path to reds dataset, if provided reds will be used for training')
    parser.add_argument('--num_workers', default=0, type=int, help='num workers in dataloaders')
    parser.add_argument('--enable_wandb', action='store_true', default=False, help='whether to enable wandb')
    parser.add_argument('--sliding_window', default=1, type=int, help='sliding window size for processing video by '
                                                                      'encoder')
    parser.add_argument('--model_type', default="vsrvc", type=str, help='trained model type, options: vsrvc, vsrvc_res')
    parser.add_argument('--vsr', action='store_true', default=False, help='whether to train VSR')
    parser.add_argument('--vc', action='store_true', default=False, help='whether to train VC')
    parser.add_argument('--log_grads', action='store_true', default=False, help='whether to log grads')
    return parser.parse_args()


def get_run_name(params):
    if params.model_type == "vsrvc":
        prefix = ('ISR' if params.vsr else '') + ('IC' if params.vc else '')
    else:
        prefix = ('VSR' if params.vsr else '') + ('VC' if params.vc else '')

    model_type = ' '
    if params.model_type == "vsrvc_res_mv":
        model_type = ' mv '
    elif params.model_type == "vsrvc_shallow":
        model_type = ' shallow '
    elif params.model_type == "vsrvc_basic":
        model_type = ' basic '

    multi_input = ' multi_input' if params.multi_input else ''
    dataset = "vimeo" if params.vimeo_path is not None else "reds"
    return f"{prefix}{model_type}{params.lmbda} {params.weighting} x{params.scale}{multi_input} {dataset}"


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    if params.model_type == "vsrvc":
        f = vsrvc
    elif params.model_type == "vsrvc_res_mv":
        f = vsrvc_motion_residual
    elif params.model_type == "vsrvc_shallow":
        f = vsrvc_shallow_encoder
    elif params.model_type == "vsrvc_basic":
        f = vsrvc_basic
    else:
        raise ValueError("Unrecognized model_type. Supported ones are: vsrvc, vsrvc_res")
    train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type = f(params, kwargs)
    run_name = get_run_name(params)
    params.save_path = os.path.join(params.save_path, run_name)
    if params.multi_input:
        train_dataloader = {
            key: DataLoader(
                    training_set,
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=params.num_workers,
                    pin_memory=True,
                    drop_last=True,) for key, training_set in train_set.items()}
        test_dataloader = {
            key: DataLoader(
                testing_set,
                batch_size=params.batch_size,
                shuffle=False,
                num_workers=params.num_workers,
                pin_memory=True,
                drop_last=True, ) for key, testing_set in test_set.items()}
    else:
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
    task_dict: Dict[str, dict] = {}
    if params.vc:
        task_dict["vc"] = {'metrics': ['psnr', 'ssim', 'bpp'],
                           'metrics_fn': CompressionTaskMetrics(),
                           'loss_fn': RateDistortionLoss(params.lmbda),
                           'weight': [1, 1, 0]}
    else:
        task_dict["vc"] = {'metrics': [], 'metrics_fn': DummyMetrics(), 'loss_fn': DummyLoss(), 'weight': []}
    if params.vsr:
        task_dict["vsr"] = {'metrics': ['psnr', 'ssim'],
                            'metrics_fn': QualityMetrics(),
                            'loss_fn': VSRLoss(params.lmbda),
                            'weight': [1, 1]}
    else:
        task_dict["vsr"] = {'metrics': [], 'metrics_fn': DummyMetrics(), 'loss_fn': DummyLoss(), 'weight': []}
    if params.enable_wandb:
        wandb.init(project="VSRVC", name=run_name)
    my_trainer = Trainer(task_dict=task_dict,
                         weighting=params.weighting,
                         architecture=params.arch,
                         encoder_class=encoder_class,
                         decoders=decoders,
                         decoder_kwargs=decoder_kwargs,
                         rep_grad=params.rep_grad,
                         multi_input=params.multi_input,
                         optim_param=optim_param,
                         scheduler_param=scheduler_param,
                         save_path=params.save_path,
                         load_path=params.load_path,
                         logging=params.enable_wandb,
                         print_interval=100,
                         lmbda=params.lmbda,
                         model_type=model_type,
                         scale=params.scale,
                         log_grads=params.log_grads,
                         dataset_type="vimeo" if params.vimeo_path is not None else "reds",
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
