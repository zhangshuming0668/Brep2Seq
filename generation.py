# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import CadDataset
from models.brep2seq_gan import GAN
from models.modules.module_utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser("latent_gan model")
parser.add_argument("traintest", choices=("train_gan", "generate"), help="Whether to train or generate")
parser.add_argument("--dataset", choices=("caddataset",), default="caddataset", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=12,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for generate",
)
parser.add_argument(
    "--checkpoint_gan",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for GAN",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="BrepToSeq_LGAN",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser.add_argument("--dim_n", type=int, default=32)
parser.add_argument("--dim_h", type=int, default=512)
parser.add_argument("--dim_z", type=int, default=256)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="Wasserstein_D",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_top_k=5,
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True, 
    gradient_clip_val=1.0
)

if args.dataset == "caddataset":
    Dataset = CadDataset
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train_gan":
    # Train gan net
    print(
        f"""
-----------------------------------------------------------------------------------
Latent_GAN
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = GAN(args)
    train_data = Dataset(root_dir=args.dataset_path, split="gan", center_and_scale=False) 
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
 
    trainer.fit(model, train_loader)
    
else:
    # generate fake CADmodel
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_path, split="test", center_and_scale=False)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = GAN.load_from_checkpoint(args.checkpoint_gan)
    trainer.test(model, dataloaders=[test_loader], verbose=False)
