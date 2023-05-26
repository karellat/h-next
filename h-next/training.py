import math
import os.path
import socket
from typing import Dict, Any

import click
import pytorch_lightning as pl
import torch
import wandb
from git import Repo
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torchsummary import summary
from lovely_tensors import chans
from hnets.hnet_lite import HConv2d
import matplotlib.pyplot as plt

from data import get_datamodule
from models import InvariantNet, get_model
from utils import log_wandb, ClickDictionaryType, get_all_module_of_type, make_grid
from upscalehnet import UpscaleHConv2d


@click.command()
@click.option('-n', '--name', default=None, type=str)
@click.option('-d', '--debug_run', default=False, type=bool, is_flag=True)
@click.option('-p', '--profiler_run', default=False, type=bool, is_flag=True)
@click.option('-g', '--track_grads', default=False, type=bool, is_flag=True)
@click.option('-e', '--epochs', default=10, type=int)
@click.option('--early_stopping', default=-1, type=int)
@click.option('--wandb_offline', default=True, type=bool, is_flag=True)
@click.option('--datamodule_name', default="mnist-rot-test", type=str)
@click.option('--datamodule_hparams', default=dict(batch_size=64, data_dir='/tmp', pad=0), type=ClickDictionaryType())
@click.option('--optimizer_name', default="AdamW", type=str)
@click.option('--optimizer_hparams', default=dict(lr=1e-3), type=ClickDictionaryType())
@click.option('--backbone_name', default="HnetBackbone", type=str)
@click.option('--backbone_hparams', default=dict(), type=ClickDictionaryType())
@click.option('--classnet_name', default="HnetPooling", type=str)
@click.option('--classnet_hparams', default=dict(), type=ClickDictionaryType())
@click.option('--label_smoothing', default=0.0, type=float)
@click.option('--lr_name', type=str, default="MultiStepLR")
@click.option('--lr_hparams', default=dict(milestones=[3, 6], gamma=0.1), type=ClickDictionaryType())
@click.option('--wandb_project', default="hnext", type=str)
@click.option('--run_tag', default=None, type=str)
@click.option('--seed', default=42, type=int)
def hnet_train(name: str,
               debug_run: bool,
               profiler_run: bool,
               track_grads: bool,
               epochs: int,
               wandb_offline: bool,
               early_stopping: int,
               datamodule_name: str,
               datamodule_hparams: Dict[str, Any],
               optimizer_name: str,
               optimizer_hparams: Dict[str, Any],
               backbone_name: str,
               backbone_hparams: Dict[str, Any],
               classnet_name: str,
               classnet_hparams: Dict[str, Any],
               label_smoothing: float,
               lr_name: str,
               lr_hparams: Dict[str, Any],
               wandb_project: str,
               run_tag: str,
               seed: int):
    seed_everything(seed=seed, workers=True)
    torch.set_float32_matmul_precision("high")
    # List available gpu
    for gpu_idx in range(torch.cuda.device_count()):
        logger.debug(
            f"GPU {gpu_idx} - {torch.cuda.get_device_name(gpu_idx)} ({torch.cuda.get_device_properties(gpu_idx).total_memory / 1e+6:.0f} MB)")
    # convert dictionary parameters
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if debug_run or profiler_run:
        os.environ["WANDB_MODE"] = "offline"
        if not profiler_run:
            logger.debug("Running in debug mode (small datasets, offline, etc).")
            # NOTE: Debug does not work running parallel workers
            datamodule_hparams['num_workers'] = 0
    else:
        os.environ["WANDB_MODE"] = "online"
    # wandb
    _tags = []
    if run_tag is not None:
        _tags.append(run_tag)

    wandb_logger = WandbLogger(project=wandb_project,
                               name=name,
                               tags=_tags,
                               log_model=True,
                               config=dict(
                                   backbone_name=backbone_name,
                                   backbone_hparams=backbone_hparams,
                                   classnet_name=classnet_name,
                                   classnet_hparams=classnet_hparams
                               ))

    datamodule = get_datamodule(datamodule_name)(**datamodule_hparams)
    hnet_backbone = get_model(model_name=backbone_name,
                              model_hparams=backbone_hparams)
    pred_net = get_model(model_name=classnet_name,
                         model_hparams=classnet_hparams)
    # model
    model = InvariantNet(hnet_backbone,
                         pred_net,
                         input_shape=datamodule.sample_input_shape,
                         optimizer_name=optimizer_name,
                         optimizer_hparams=optimizer_hparams,
                         label_smoothing=label_smoothing,
                         lr_name=lr_name,
                         lr_hparams=lr_hparams)

    # callbacks
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    wandb_logger.experiment.config["sha"] = sha
    wandb_logger.experiment.config["device"] = socket.gethostname()
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    wandb.define_metric('val_acc', summary='max')
    _wandb_out_status = 'aborted'

    # train
    if debug_run:
        _debug_params = dict(
            limit_train_batches=0.125,
            limit_val_batches=0.125,
            limit_test_batches=0.125,
            detect_anomaly=True,
            deterministic="warn",
        )
    else:
        _debug_params = dict(
            detect_anomaly=False,
            deterministic=False,
        )
    _callbacks = [lr_monitor_callback]
    if early_stopping > 0:
        _callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping))

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "auto",
                         devices=-1 if torch.cuda.is_available() else "auto",
                         logger=wandb_logger,
                         max_epochs=epochs,
                         track_grad_norm=2 if track_grads else -1,
                         **_debug_params,
                         enable_model_summary=False,
                         profiler=None if not profiler_run else PyTorchProfiler(),
                         callbacks=_callbacks
                         )

    try:
        _ = summary(model,
                    input_data=datamodule.sample_input_shape[1:],
                    verbose=True)
        # Fitting
        trainer.fit(model=model,
                    datamodule=datamodule)
        _wandb_out_status = "success"
    finally:
        wandb_logger.finalize(_wandb_out_status)


if __name__ == "__main__":
    hnet_train()
