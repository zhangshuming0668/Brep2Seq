# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.brep2seq import BreptoSeq
from data.dataset import CadDataset

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser("Brep2Seq reconstruction")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="BrepToSeq",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--attention_dropout", type=float, default=0.1)
parser.add_argument("--act-dropout", type=float, default=0.1)

parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_z", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
parser.add_argument("--dim_feedforward", type=int, default=256)
parser.add_argument("--n_layers_encode", type=int, default=8)
parser.add_argument("--n_layers_decode", type=int, default=8)
parser.add_argument("--similarity_loss", choices=("dann", "mmd"), default="dann", help="method for similarity loss function")

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_top_k=10,
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

if args.traintest == "train":
    # Train/val
    print(
        f"""
-----------------------------------------------------------------------------------
Brep2Seq Network
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = BreptoSeq(args)
    train_data = CadDataset(root_dir=args.dataset_path, split="train")
    val_data = CadDataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers
    )    
    trainer.fit(model, train_loader, val_loader)
    
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    model = BreptoSeq.load_from_checkpoint(args.checkpoint)
    test_data = CadDataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)
    
