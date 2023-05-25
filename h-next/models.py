import math
import sys
from collections import OrderedDict
from typing import Dict, Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
import wandb
from einops import repeat, rearrange
from einops.layers.torch import Rearrange, Reduce
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.states import RunningStage
from torch import optim, nn
from torch.optim import lr_scheduler

from data import TEST_INVARIANCE_KEY
from hnets.hnet_lite import HConv2d, HView, HBatchNorm, HNonlin, HSumMagnitutes, HMeanPool, HStackMagnitudes
from invpool import get_single_invariant_layer, MultipleZernikePooling, ParallelZernikePooling
from upscalehnet import UpscaleHConv2d
from utils import cbar_plot


def get_model(model_name: str,
              model_hparams: Dict[str, Any]):
    if hasattr(sys.modules[__name__], model_name):
        return getattr(sys.modules[__name__], model_name)(**model_hparams)
    else:
        raise RuntimeError(f"Unknown model: {model_name}")


class InvariantNet(pl.LightningModule):
    def __init__(self,
                 backbone_network: nn.Module,
                 classification_network: nn.Module,
                 input_shape: List[int],
                 optimizer_name: str = "SGD",
                 optimizer_hparams: Dict[str, Any] = dict(lr=1e-3),
                 lr_name: str = "MultiStepLR",  # None
                 lr_hparams: Dict[str, Any] = dict(milestones=[3, 6, 9], gamma=0.1),
                 label_smoothing=0,
                 log_invariance=True,
                 ):  # MultiStep
        super().__init__()
        self.save_hyperparameters(ignore=['backbone_network', 'classification_network', 'log_invariance'])

        self.backbone_network = backbone_network
        self.log_invariance = log_invariance
        self.classification_network = classification_network
        self.loss_fnc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=10, normalize='all')
        self.example_input_array = torch.Tensor(*input_shape)

    @staticmethod
    def _unpack_batch(batch):
        if type(batch) is dict:
            return batch['image'], batch['label']
        elif (type(batch) is list) or (type(batch) is tuple):
            return batch
        else:
            raise NotImplementedError(f"Unknown batch type {type(batch)}")

    def forward(self, x):
        x_ = self.backbone_network(x)
        return self.classification_network(x_)

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if type(batch) is dict:
            assert 'val' in batch.keys(), "Multiple validation datasets has to contains \"val\" dataset"
            for k, v in batch.items():
                preds, loss, acc = self._get_preds_loss_accuracy(v[0])
                if k == 'val':
                    res_preds = preds
                    res_loss = loss
                    _, res_targets = self._unpack_batch(v[0])
                self.log(f'{k}_loss', loss)
                self.log(f'{k}_acc', acc)
            return {
                'loss': res_loss,
                'preds': res_preds,
                'target': res_targets}
        else:
            preds, loss, acc = self._get_preds_loss_accuracy(batch)
            _, targets = self._unpack_batch(batch)
            self.log('val_loss', loss)
            self.log('val_acc', acc)

            return {'loss': loss,
                    'preds': preds,
                    'target': targets}

    def test_step(self, batch, batch_idx):
        # Log the first batch
        if type(batch) is dict:
            assert TEST_INVARIANCE_KEY in batch.keys(), "Expecting validation dataset rotated"
            if (batch_idx == 0) and self.log_invariance:
                self._test_invariance_backbone(self._unpack_batch(batch[TEST_INVARIANCE_KEY][0]))
            for k, v in batch.items():
                preds, loss, acc = self._get_preds_loss_accuracy(v[0])
                self.log(f'{k}_loss', loss)
                self.log(f'{k}_acc', acc)

        else:
            if (batch_idx == 0) and self.log_invariance:
                self._test_invariance_backbone(self._unpack_batch(batch))
            preds, loss, acc = self._get_preds_loss_accuracy(batch)
            self.log('test_loss', loss)
            self.log('test_acc', acc)

    def configure_optimizers(self):
        if hasattr(optim, self.hparams.optimizer_name):
            optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(),
                                                                    **self.hparams.optimizer_hparams)
        else:
            raise RuntimeError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')
        if self.hparams.lr_name == "None":
            return optimizer
        elif hasattr(lr_scheduler, self.hparams.lr_name):
            scheduler = getattr(lr_scheduler, self.hparams.lr_name)(optimizer, **self.hparams.lr_hparams)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        else:
            raise RuntimeError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

    def _get_preds_loss_accuracy(self, batch):
        imgs, labels = self._unpack_batch(batch)
        features = self.backbone_network(imgs)
        preds = self.classification_network(features)
        loss = self.loss_fnc(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        return preds, loss, acc

    @property
    def _wandb_logger(self) -> WandbLogger:
        return self.logger

    def validation_epoch_end(self, outputs):
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return
        matplotlib.use('Agg')
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        confusion_matrix = self.confusion_matrix(preds, targets)

        df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index=range(10), columns=range(10))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        self._wandb_logger.experiment.log({"Confusion matrix": wandb.Image(fig_)})
        plt.close(fig_)

    def _test_invariance_backbone(self, batch):
        # TODO: Put it out of the model definition
        logger.debug("Drawing channels of the best model.")
        imgs, labels = batch
        unique_labels, unique_labels_idx = np.unique(labels.detach().cpu().numpy(), return_index=True)
        imgs = imgs[unique_labels_idx, ...]
        labels = labels[unique_labels_idx]
        if len(unique_labels_idx) != 10:
            logger.warning(f"Some mnist number missing, {unique_labels}")
        rotated_imgs = torch.rot90(imgs, k=1, dims=[-2, -1])

        feature_extractor = self.backbone_network
        feature_extractor.eval()
        with torch.no_grad():
            features = feature_extractor(imgs)
            rotated_features = feature_extractor(rotated_imgs)

        # Visualize difference
        matplotlib.use('Agg')
        for img_id in range(imgs.shape[0]):
            n_rows = features.shape[-1]
            fig, axs = plt.subplots(n_rows, 6, figsize=(15, n_rows * 2))
            for filter_id in range(n_rows):
                (ax0, ax1, ax2, ax3, ax4, ax5) = axs[filter_id]
                sample_filter = features[img_id, :, :, filter_id].cpu().numpy()
                sample_rotated_filter = torch.rot90(rotated_features[img_id, :, :, filter_id],
                                                    k=-1,
                                                    dims=[-2, -1]).cpu().numpy()
                difference = sample_rotated_filter - sample_filter
                var_norm = sample_filter / (torch.clamp(torch.var(features[img_id, :, :, :]), min=1e-7)).cpu().numpy()
                cbar_plot(fig, ax0, np.isclose(sample_filter, 0.0), cmap="Greys")
                cbar_plot(fig, ax1, sample_filter)
                cbar_plot(fig, ax2, var_norm)
                cbar_plot(fig, ax3, sample_rotated_filter)
                cbar_plot(fig, ax4, difference)
                ax5.boxplot(np.abs(difference.flatten()), showfliers=True)
            (ax0, ax1, ax2, ax3, ax4, ax5) = axs[0]
            ax0.set_title('Zero mask')
            ax1.set_title('Non-rotated feature')
            ax2.set_title('Layer variance norm')
            ax3.set_title('Rotated feature')
            ax4.set_title('Feature difference')
            ax5.set_title('Absolute error distribution')
            fig.tight_layout()
            fig.suptitle(f'Label {labels[img_id]}')
            self._wandb_logger.experiment.log({f'Hnet features': wandb.Image(fig)}, commit=True)
            plt.close('all')


class HnetBackbone(nn.Module):
    def __init__(self,
                 nf1=8,
                 nf2=16,
                 nf3=32,
                 maximum_order=1,
                 activation_fnc=nn.functional.relu,
                 h_meanpool: bool = True,
                 phase_offset: bool = True,
                 bias: bool = True,
                 kernel_size: int = 5,
                 batch_norm: bool = True,
                 batchnorm_affine: bool = True,
                 n_rings: int = 4):

        super().__init__()
        self.view = HView()
        self.hblock1 = HnetBackbone._hconv_block(name_id=1,
                                                 in_channels=1,
                                                 out_channels=nf1,
                                                 in_max_order=0,
                                                 out_max_order=maximum_order,
                                                 h_meanpool=h_meanpool,
                                                 phase_offset=phase_offset,
                                                 kernel_size=kernel_size,
                                                 n_rings=n_rings,
                                                 batch_norm=batch_norm,
                                                 batchnorm_affine=batchnorm_affine,
                                                 act=activation_fnc,
                                                 bias=bias)
        self.hblock2 = HnetBackbone._hconv_block(name_id=2,
                                                 in_channels=nf1,
                                                 out_channels=nf2,
                                                 in_max_order=maximum_order,
                                                 out_max_order=maximum_order,
                                                 h_meanpool=h_meanpool,
                                                 phase_offset=phase_offset,
                                                 kernel_size=kernel_size,
                                                 batch_norm=batch_norm,
                                                 batchnorm_affine=batchnorm_affine,
                                                 n_rings=n_rings,
                                                 act=activation_fnc,
                                                 bias=bias)
        self.hblock3 = HnetBackbone._hconv_block(name_id=3,
                                                 in_channels=nf2,
                                                 out_channels=nf3,
                                                 in_max_order=maximum_order,
                                                 out_max_order=maximum_order,
                                                 h_meanpool=False,
                                                 phase_offset=phase_offset,
                                                 kernel_size=kernel_size,
                                                 batch_norm=batch_norm,
                                                 batchnorm_affine=batchnorm_affine,
                                                 n_rings=n_rings,
                                                 act=activation_fnc,
                                                 bias=bias)
        self.last_hconv = HConv2d(nf3, 10,
                                  kernel_size,
                                  padding=(kernel_size - 1) // 2,
                                  n_rings=n_rings,
                                  phase=phase_offset,
                                  in_max_order=maximum_order,
                                  out_max_order=0)
        self.hsum_pool = HSumMagnitutes(keep_dims=False)

    def forward(self, x):
        x_ = self.view(x)
        x_ = self.hblock1(x_)
        x_ = self.hblock2(x_)
        x_ = self.hblock3(x_)
        x_ = self.last_hconv(x_)
        x_ = self.hsum_pool(x_)
        return x_

    @classmethod
    def _hconv_block(cls,
                     name_id: int,
                     in_channels: int,
                     out_channels: int,
                     in_max_order: int,
                     out_max_order: int,
                     act=nn.functional.relu,
                     h_meanpool: bool = True,
                     phase_offset: bool = True,
                     bias: bool = True,
                     batch_norm: bool = True,
                     kernel_size: int = 5,
                     batchnorm_affine: bool = True,
                     n_rings: int = 4):
        layers = [(f"{name_id}hconv0", HConv2d(in_channels, out_channels, kernel_size,
                                               in_max_order=in_max_order,
                                               out_max_order=out_max_order,
                                               padding=(kernel_size - 1) // 2,
                                               n_rings=n_rings,
                                               phase=phase_offset)),
                  (f"{name_id}act0", HNonlin(act,
                                             max_order=out_max_order,
                                             channels=out_channels,
                                             eps=1e-12,
                                             bias=bias)),
                  (f"{name_id}hconv1",
                   HConv2d(out_channels,
                           out_channels,
                           kernel_size,
                           in_max_order=out_max_order,  # First of block changes order
                           out_max_order=out_max_order,
                           padding=(kernel_size - 1) // 2,
                           n_rings=n_rings,
                           phase=phase_offset))]
        if batch_norm:
            layers.append((f"{name_id}hbn", HBatchNorm(out_channels, affine=batchnorm_affine)))
        if h_meanpool:
            layers.append((f"{name_id}hmeanpool", HMeanPool(kernel_size=(2, 2), strides=(2, 2))))

        return nn.Sequential(OrderedDict(layers))


class UpscaleHnetBackbone(nn.Module):
    def __init__(self,
                 nf1=8,
                 nf2=16,
                 nf3=32,
                 out_channels: int = 10,
                 maximum_order=1,
                 in_channels=1,
                 activation_fnc="gelu",
                 h_meanpool: bool = True,
                 phase_offset: bool = True,
                 bias: bool = True,
                 kernel_size: int = 15,
                 batchnorm_affine: bool = True,
                 n_rings: int = 3,
                 scale_factor: int = 2,
                 scale_mode='bilinear',
                 input_shape=(32, 32),
                 circular_masking=False,
                 normalization='BN',  # BN for batch norm, LN for layer norm, None for no norm
                 _test_phase_norm=False,
                 ):

        super().__init__()
        self.view = HView()
        assert hasattr(nn.functional, activation_fnc), "Unknown activation function"
        channel_height, channel_width = input_shape
        channel_height *= scale_factor
        channel_width *= scale_factor
        activation_fnc = getattr(nn.functional, activation_fnc)
        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode=scale_mode)

        self.hblock1 = UpscaleHnetBackbone._up_hconv_block(name_id=1,
                                                           in_height=channel_height,
                                                           in_width=channel_width,
                                                           in_channels=in_channels,
                                                           out_channels=nf1,
                                                           in_max_order=0,
                                                           out_max_order=maximum_order,
                                                           h_meanpool=h_meanpool,
                                                           phase_offset=phase_offset,
                                                           kernel_size=kernel_size,
                                                           n_rings=n_rings,
                                                           scale_factor=scale_factor,
                                                           norm_fnc=normalization,
                                                           norm_affine=batchnorm_affine,
                                                           circular_masking=circular_masking,
                                                           act=activation_fnc,
                                                           _test_phase_norm=_test_phase_norm,
                                                           bias=bias)
        self.hblock2 = UpscaleHnetBackbone._up_hconv_block(name_id=2,
                                                           in_height=int(channel_height / 2),
                                                           in_width=int(channel_width / 2),
                                                           in_channels=nf1,
                                                           out_channels=nf2,
                                                           in_max_order=maximum_order,
                                                           out_max_order=maximum_order,
                                                           h_meanpool=h_meanpool,
                                                           phase_offset=phase_offset,
                                                           kernel_size=kernel_size,
                                                           scale_factor=scale_factor,
                                                           norm_fnc=normalization,
                                                           norm_affine=batchnorm_affine,
                                                           n_rings=n_rings,
                                                           circular_masking=circular_masking,
                                                           _test_phase_norm=_test_phase_norm,
                                                           act=activation_fnc,
                                                           bias=bias)
        self.hblock3 = UpscaleHnetBackbone._up_hconv_block(name_id=3,
                                                           in_height=int(channel_height / 4),
                                                           in_width=int(channel_width / 4),
                                                           in_channels=nf2,
                                                           out_channels=nf3,
                                                           in_max_order=maximum_order,
                                                           out_max_order=maximum_order,
                                                           h_meanpool=False,
                                                           phase_offset=phase_offset,
                                                           kernel_size=kernel_size,
                                                           scale_factor=scale_factor,
                                                           norm_fnc=normalization,
                                                           norm_affine=batchnorm_affine,
                                                           n_rings=n_rings,
                                                           _test_phase_norm=_test_phase_norm,
                                                           circular_masking=circular_masking,
                                                           act=activation_fnc,
                                                           bias=bias)
        self.last_hconv = UpscaleHConv2d(nf3, out_channels,
                                         kernel_size,
                                         padding=(kernel_size - 1) // 2,
                                         n_rings=n_rings,
                                         phase=phase_offset,
                                         circular_masking=circular_masking,
                                         mask_shape=int(channel_width / 4),
                                         norm_polar2cart=_test_phase_norm,
                                         in_max_order=maximum_order,
                                         out_max_order=0)
        self.hsum_pool = HSumMagnitutes(keep_dims=False)

    def forward(self, x):
        x_ = self.upsample(x)
        x_ = self.view(x_)
        x_ = self.hblock1(x_)
        x_ = self.hblock2(x_)
        x_ = self.hblock3(x_)
        x_ = self.last_hconv(x_)
        x_ = self.hsum_pool(x_)
        return x_

    @classmethod
    def _up_hconv_block(cls,
                        name_id: int,
                        in_height: int,
                        in_width: int,
                        in_channels: int,
                        out_channels: int,
                        in_max_order: int,
                        out_max_order: int,
                        scale_factor: int,
                        act=nn.functional.relu,
                        h_meanpool: bool = True,
                        phase_offset: bool = True,
                        bias: bool = True,
                        batch_norm: bool = True,
                        circular_masking: bool = False,
                        kernel_size: int = 5,
                        norm_affine: bool = True,
                        _test_phase_norm=False,
                        norm_fnc=None,
                        n_rings: int = 4):
        layers = [(f"{name_id}_up_hconv0", UpscaleHConv2d(in_channels, out_channels, kernel_size,
                                                          in_max_order=in_max_order,
                                                          out_max_order=out_max_order,
                                                          circular_masking=circular_masking,
                                                          mask_shape=in_height,
                                                          norm_polar2cart=_test_phase_norm,
                                                          padding=(kernel_size - 1) // 2,
                                                          n_rings=n_rings,
                                                          phase=phase_offset)),
                  (f"{name_id}act0", HNonlin(act,
                                             max_order=out_max_order,
                                             channels=out_channels,
                                             eps=1e-12,
                                             bias=bias)),
                  (f"{name_id}_up_hconv1",
                   UpscaleHConv2d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  in_max_order=out_max_order,  # First of block changes order
                                  out_max_order=out_max_order,
                                  circular_masking=circular_masking,
                                  mask_shape=in_height,
                                  norm_polar2cart=_test_phase_norm,
                                  padding=(kernel_size - 1) // 2,
                                  n_rings=n_rings,
                                  phase=phase_offset))]
        if norm_fnc == 'BN':
            layers.append((f"{name_id}hbn", HBatchNorm(out_channels, affine=norm_affine)))
        elif norm_fnc is None or norm_fnc == 'None':
            pass
        else:
            raise NotImplementedError(f"Unknown normalization, {norm_fnc}")

        if h_meanpool:
            layers.append((f"{name_id}hmeanpool", HMeanPool(kernel_size=(2, 2),
                                                            strides=(2, 2))))

        return nn.Sequential(OrderedDict(layers))


class UpscaleHnetWideBackbone(nn.Module):
    def __init__(self,
                 model_str="B-8-MP,B-16-MP,B-32-MP,B-64",
                 maximum_order=2,
                 in_channels=3,
                 activation_fnc="relu",
                 h_meanpool: bool = True,
                 phase_offset: bool = True,
                 bias: bool = True,
                 kernel_size: int = 15,
                 batchnorm_affine: bool = True,
                 n_rings: int = 3,
                 scale_factor: int = 2,
                 scale_mode='bilinear',
                 input_shape=(32, 32),
                 circular_masking=True,
                 normalization='BN',  # BN for batch norm, LN for layer norm, None for no norm
                 _test_phase_norm=False,
                 ):

        super().__init__()
        self.view = HView()
        assert hasattr(nn.functional, activation_fnc), "Unknown activation function"
        channel_height, channel_width = input_shape
        channel_height *= scale_factor
        channel_width *= scale_factor
        activation_fnc = getattr(nn.functional, activation_fnc)
        self.upsample = nn.Upsample(scale_factor=scale_factor,
                                    mode=scale_mode)

        _last_out_channels = in_channels
        block_layers = []
        _in_max_order = 0
        _last_channel_size = channel_height
        for idx, layer in enumerate(model_str.split(',')):
            params = layer.upper().split('-')
            block_layers.append(UpscaleHnetWideBackbone._up_hconv_block(name_id=idx,
                                                                        in_height=_last_channel_size,
                                                                        in_width=_last_channel_size,
                                                                        in_channels=_last_out_channels,
                                                                        out_channels=int(params[1]),
                                                                        in_max_order=_in_max_order,
                                                                        out_max_order=maximum_order,
                                                                        h_meanpool=True if "MP" in params else False,
                                                                        phase_offset=phase_offset,
                                                                        kernel_size=kernel_size,
                                                                        n_rings=n_rings,
                                                                        scale_factor=scale_factor,
                                                                        norm_fnc=normalization,
                                                                        norm_affine=batchnorm_affine,
                                                                        circular_masking=circular_masking,
                                                                        act=activation_fnc,
                                                                        _test_phase_norm=_test_phase_norm,
                                                                        bias=bias))
            _in_max_order = maximum_order
            _last_out_channels = int(params[1])
            if "MP" in params:
                assert _last_channel_size % 2 == 0
                _last_channel_size = int(_last_channel_size / 2)
        self.blocks = nn.Sequential(*block_layers)
        self.stack_layer = HStackMagnitudes(keep_dims=False)

    def forward(self, x):
        x_ = self.upsample(x)
        x_ = self.view(x_)
        x_ = self.blocks(x_)
        x_ = self.stack_layer(x_)

        return torch.flatten(input=x_,
                             start_dim=-2,
                             end_dim=-1)

    @classmethod
    def _up_hconv_block(cls,
                        name_id: int,
                        in_height: int,
                        in_width: int,
                        in_channels: int,
                        out_channels: int,
                        in_max_order: int,
                        out_max_order: int,
                        scale_factor: int,
                        act=nn.functional.relu,
                        h_meanpool: bool = True,
                        phase_offset: bool = True,
                        bias: bool = True,
                        batch_norm: bool = True,
                        circular_masking: bool = False,
                        kernel_size: int = 5,
                        norm_affine: bool = True,
                        _test_phase_norm=False,
                        norm_fnc=None,
                        n_rings: int = 4):
        layers = [(f"{name_id}uphconv0", UpscaleHConv2d(in_channels, out_channels, kernel_size,
                                                        in_max_order=in_max_order,
                                                        out_max_order=out_max_order,
                                                        circular_masking=circular_masking,
                                                        mask_shape=in_height,
                                                        norm_polar2cart=_test_phase_norm,
                                                        padding=(kernel_size - 1) // 2,
                                                        n_rings=n_rings,
                                                        phase=phase_offset)),
                  (f"{name_id}act0", HNonlin(act,
                                             max_order=out_max_order,
                                             channels=out_channels,
                                             eps=1e-12,
                                             bias=bias)),
                  (f"{name_id}uphconv1",
                   UpscaleHConv2d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  in_max_order=out_max_order,  # First of block changes order
                                  out_max_order=out_max_order,
                                  circular_masking=circular_masking,
                                  mask_shape=in_height,
                                  norm_polar2cart=_test_phase_norm,
                                  padding=(kernel_size - 1) // 2,
                                  n_rings=n_rings,
                                  phase=phase_offset)),
                  ]
        if norm_fnc == 'BN':
            layers.append((f"{name_id}hbn1", HBatchNorm(out_channels, affine=norm_affine)))
        elif norm_fnc is None or norm_fnc == 'None':
            pass
        else:
            raise NotImplementedError(f"Unknown normalization, {norm_fnc}")

        if h_meanpool:
            layers.append((f"{name_id}hmeanpool", HMeanPool(kernel_size=(2, 2),
                                                            strides=(2, 2))))

        return nn.Sequential(OrderedDict(layers))


class HnetPooling(torch.nn.Module):
    def __init__(self, ncl: int = 10, channels_dim=(1, 2)):
        super(HnetPooling, self).__init__()
        self.bias = torch.ones(ncl) * 1e-2
        self.bias = self.bias.type(torch.get_default_dtype())
        self.channels_dim = channels_dim
        self.bias = torch.nn.Parameter(self.bias)
        self.__ncl = ncl

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: [Batch Size, Height, Width, Channels]

        Returns
        -------

        """
        if x.shape[3] != self.__ncl or x.ndim != 4:
            logger.warning(f"""Expecting shape [Batch, Height, Width, Channels], go {x.shape}""")
        return torch.mean(x, dim=self.channels_dim) + self.bias.view(1, -1)


class LastInvPool(torch.nn.Module):
    def __init__(self,
                 invariant_name="M00",
                 input_shape=(-1, 28, 28, 10),
                 channels_dim=(1, 2),
                 norm_var=True,
                 bias=True,
                 min_coord=-1,
                 max_coord=1,
                 number_classes=10,
                 eps=1e-7):
        super().__init__()
        assert input_shape[channels_dim[0]] == input_shape[
            channels_dim[1]], f"Not a square input {input_shape} in {channels_dim}"
        inv_kwargs = dict(
            min_coord=min_coord,
            max_coord=max_coord,
            number_samples=input_shape[channels_dim[0]],
            input_shape=input_shape,
            channels_dim=channels_dim
        )
        self.inv_pool = get_single_invariant_layer(invariant_name, **inv_kwargs)
        self.has_bias = bias
        if self.has_bias:
            self.bias = torch.ones(number_classes) * 1e-2
            self.bias = self.bias.type(torch.get_default_dtype())
            self.bias = torch.nn.Parameter(self.bias, requires_grad=True)
        self.norm_var = norm_var
        self._norm_dims = [*channels_dim, -1]
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalization
        if self.norm_var:
            x_ = x / (torch.clamp(torch.var(x, dim=self._norm_dims, keepdim=True), min=self._eps))
        else:
            x_ = x

        if self.has_bias:
            return self.inv_pool(x_) + self.bias.view(1, -1)
        else:
            return self.inv_pool(x_)


class MultiplePooling(torch.nn.Module):
    """
    Multiple Pooling layer
    """

    def __init__(self,
                 input_shape=(-1, 28, 28, 10),
                 channels_dim=(1, 2),
                 min_coord=-1,
                 max_coord=1,
                 variance_norm=True,
                 invariants=None,
                 dropout=False,
                 dropout_p=0.1,
                 fc_out=1,
                 keep_dim=False,
                 eps=1e-7):
        super().__init__()
        assert keep_dim or (not keep_dim and (fc_out == 1))
        if invariants is None:
            invariants = ["HuInvariant1"]
        assert input_shape[channels_dim[0]] == input_shape[
            channels_dim[1]], f"Not a square input {input_shape} in {channels_dim}"
        inv_kwargs = dict(
            min_coord=min_coord,
            max_coord=max_coord,
            number_samples=input_shape[channels_dim[0]],
            input_shape=input_shape,
            channels_dim=channels_dim
        )
        self.keep_dim = keep_dim
        self.invariants = torch.nn.ModuleList(
            [get_single_invariant_layer(inv_name, **inv_kwargs) for inv_name in invariants]
        )
        self.mean_pool = get_single_invariant_layer("GlobalMeanPool", **inv_kwargs, )
        self.variance_norm = variance_norm
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = torch.nn.Dropout1d(p=dropout_p)
        self.eps = eps
        self.shared_fc = torch.nn.Linear(in_features=1 + len(self.invariants), out_features=fc_out)

    def forward(self, x: torch.Tensor):
        mp_x = self.mean_pool(x)[..., None, :]
        if self.variance_norm:
            _x = x / (torch.clamp(torch.var(x, dim=[1, 2], keepdim=True), min=self.eps))
        else:
            _x = x
        inv_x = [invariant(_x)[..., None, :] for invariant in self.invariants]
        _x = torch.cat([mp_x, *inv_x], dim=1)
        if self.dropout:
            _x = self.dropout_layer(_x)
        _x = _x.permute([0, 2, 1])
        _x = self.shared_fc(_x)
        if not self.keep_dim:
            _x = _x[..., 0]
        return _x


class ZernikeProtypePooling(torch.nn.Module):
    _layer_sep = ','
    _param_sep = '-'

    class MeanPoolLayer(torch.nn.Module):
        def __init__(self, channels_dim):
            super().__init__()
            self.channels_dim = channels_dim

        def forward(self, x: torch.Tensor):
            return torch.mean(x, dim=self.channels_dim)

    class SharedLinearLayer(torch.nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.shared_ln = torch.nn.Conv2d(in_channels=1,
                                             out_channels=1,
                                             kernel_size=(in_features, 1),
                                             padding='valid')

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.shared_ln(x[:, None])[:, 0]

    class PermuteLayer(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.permute([0, 3, 1, 2])

    @staticmethod
    def _parse_multiple_pooling(layer_str: str,
                                input_shape: List[int]) -> torch.nn.Module:
        # ZM[max rank],[mean_pool],[no_variance]
        # example "ZM3,MP,NOVAR
        assert input_shape[2] == input_shape[3]
        input_size = input_shape[3]
        params = layer_str[3:].upper().split(ZernikeProtypePooling._param_sep)
        max_rank = int(params[0])
        variance_norm = False if 'NOVAR' in params else True
        mean_pool = True if 'MP' in params else False
        return MultipleZernikePooling(input_size=input_size,
                                      max_rank=max_rank,
                                      variance_norm=variance_norm,
                                      mean_pool=mean_pool,
                                      spatial_dim=[2, 3])

    @staticmethod
    def _parse_paralell_pooling(layer_str: str,
                                input_shape: List[int]) -> torch.nn.Module:
        # PZM[Number_Invarinats],[mean_pool],[no_variance]
        # example "PZM3,MP,NOVAR
        assert input_shape[2] == input_shape[3]
        input_size = input_shape[3]
        params = layer_str[4:].upper().split(ZernikeProtypePooling._param_sep)
        number_invariants = int(params[0])
        variance_norm = False if 'NOVAR' in params else True
        mean_pool = True if 'MP' in params else False
        return ParallelZernikePooling(input_size=input_size,
                                      number_invariants=number_invariants,
                                      variance_norm=variance_norm,
                                      mean_pool=mean_pool,
                                      spatial_dim=[2, 3])

    @staticmethod
    def _parse_linear_layer(layer_str: str, input_shape):
        params = layer_str[2:].upper().split(ZernikeProtypePooling._param_sep)
        out_features = int(params[0])
        bias = True if 'NOBIAS' in params else False
        in_features = input_shape[-1]
        return torch.nn.Linear(in_features=in_features,
                               out_features=out_features,
                               bias=bias)

    def __init__(self,
                 model_str: str='MP,BN,D-25,L-10',
                 input_shape: List[int]=[-1, 8, 8, 192],
                 channels_last: bool = True):
        super().__init__()
        layers = [] if not channels_last else [ZernikeProtypePooling.PermuteLayer()]

        for layer_str in model_str.split(sep=ZernikeProtypePooling._layer_sep):
            with torch.no_grad():
                _model = torch.nn.Sequential(*layers)
                last_shape = [-1,
                              *_model(torch.ones([32, *input_shape[1:]], dtype=torch.get_default_dtype())).shape[1:]]
            # NOTE: Python 3.10 add support for switch case
            if layer_str.startswith("ZM"):
                layers.append(ZernikeProtypePooling._parse_multiple_pooling(layer_str,
                                                                            input_shape=last_shape))
            elif layer_str.startswith("PZM"):
                layers.append(ZernikeProtypePooling._parse_paralell_pooling(layer_str,
                                                                            input_shape=last_shape))
            elif layer_str.startswith("F"):
                layers.append(torch.nn.Flatten(start_dim=1, end_dim=2))
            elif layer_str.startswith("A"):
                params = layer_str[2:].split(ZernikeProtypePooling._param_sep)
                assert hasattr(torch.nn, params[0])
                layers.append(getattr(torch.nn, params[0])())
            elif layer_str.startswith("LN"):
                layers.append(torch.nn.LayerNorm(normalized_shape=last_shape[1]))
            elif layer_str.startswith("L"):
                layers.append(ZernikeProtypePooling._parse_linear_layer(layer_str,
                                                                        input_shape=last_shape))
            elif layer_str.startswith("BN"):
                layers.append(torch.nn.BatchNorm1d(num_features=last_shape[1]))
            elif layer_str.startswith("MP"):
                layers.append(ZernikeProtypePooling.MeanPoolLayer(channels_dim=[2, 3]))
            elif layer_str.startswith("SL"):
                layers.append(ZernikeProtypePooling.SharedLinearLayer(in_features=last_shape[1]))
            elif layer_str.startswith("D"):
                params = layer_str[2:].upper().split(ZernikeProtypePooling._param_sep)
                layers.append(torch.nn.Dropout1d(int(params[0]) / 100))
            elif layer_str.startswith("UP"):
                params = layer_str[3:].upper().split(ZernikeProtypePooling._param_sep)
                layers.append(torch.nn.Upsample(scale_factor=int(params[0]),
                                                mode='bilinear'))
            # elif layer_str.startwith("SA"):
            #    layers.append()
            else:
                raise NotImplementedError(f"Unknown layer token \"{layer_str}\"")

        self.inv_pool = torch.nn.Sequential(*layers[0:2])
        self.classifier = torch.nn.Sequential(*layers[2:])
        self.name = model_str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.inv_pool(x)
        return self.classifier(_x)

    def string(self) -> str:
        return self.name


class ZernikeRankPooling(torch.nn.Module):

    def __init__(self, max_rank: int = 2, channel_per_rank: int = 32, input_size: int = 16, use_center_of_mass:bool=True):
        super().__init__()
        self.permute_layer = Rearrange('b h w c -> b c h w')
        self.zernike_pooling = MultipleZernikePooling(input_size=input_size,
                                                      max_rank=3,
                                                      use_center_of_mass=use_center_of_mass,
                                                      mean_pool=True,
                                                      spatial_dim=[2, 3])

        self.channels_per_rank = channel_per_rank
        self.rank_flat_layer = Rearrange('b r c -> b (r c)')
        features = (max_rank + 1) * channel_per_rank
        self.bn_layer = torch.nn.BatchNorm1d(num_features=self.zernike_pooling.number_invariants * features)
        self.flat_layer = Reduce('b n e -> b e', reduction='mean'),
        self.linear_layer = torch.nn.Linear(in_features=self.zernike_pooling.number_invariants * features,
                                            out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.unflatten(x, dim=-1, sizes=[self.max_rank + 1, self.channels_per_rank])
        x_ = torch.stack(dim=1, tensors=[self.zernike_pooling(
            self.permute_layer(x_[:, :, :, i, :])
        ) for i in range(self.max_rank + 1)])
        x_ = self.rank_flat_layer(x_)
        x_ = self.bn_layer(x_)
        x_ = self.linear_layer(x_)
        return x_

    def string(self) -> str:
        return self.name



class RelativeAttention(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 image_size,
                 dim_head,
                 heads=1,
                 dropout=0.):
        # NOTE: Based on https://github.com/chinhsuanwu/coatnet-pytorch/blob/d3ef1c3e4d6dfcc0b5f731e46774885686062452/coatnet.py#L121
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size
        self.diag = math.ceil(((self.ih - 1) ** 2 + (self.iw - 1) ** 2) ** 0.5)
        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias based on distance from middle
        # circlular positional embeddings
        self.relative_bias_table = nn.Parameter(
            torch.zeros(self.diag + 1, heads))
        coords = torch.meshgrid((torch.arange(self.ih),
                                 torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_distances = torch.sqrt(relative_coords[0] ** 2 + relative_coords[1] ** 2)
        relative_distances = torch.ceil(relative_distances)
        relative_index = relative_distances.type(torch.int64).flatten()

        # relative index order patch to relative bias table
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, repeat(self.relative_index, 'c -> c h', h=self.heads))

        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PreNorm(nn.Module):
    # https://github.com/chinhsuanwu/coatnet-pytorch/blob/d3ef1c3e4d6dfcc0b5f731e46774885686062452/coatnet.py#L17
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # https://github.com/chinhsuanwu/coatnet-pytorch/blob/d3ef1c3e4d6dfcc0b5f731e46774885686062452/coatnet.py#L45
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    # NOTE:https://github.com/chinhsuanwu/coatnet-pytorch/blob/d3ef1c3e4d6dfcc0b5f731e46774885686062452/coatnet.py#L164
    def __init__(self, inp, oup, image_size, heads=1, dim_head=32, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size

        self.attn = RelativeAttention(inp, oup, image_size, dim_head, heads, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b ih iw c -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b ih iw c', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b ih iw c -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b ih iw c', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        #TODO: implement downsampling https://github.com/chinhsuanwu/coatnet-pytorch/blob/d3ef1c3e4d6dfcc0b5f731e46774885686062452/coatnet.py#L193
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class TransformerPooling(nn.Module):
    def __init__(self,
                 in_shape: Tuple[int, int, int, int]=[-1, 32, 32,48],
                 num_transformers=2,
                 num_heads=4,
                 dim_head=48,
                 dropout=0.2):
        super().__init__()
        assert len(in_shape) == 4, "Expecting in shape in format [B, H, W, C]"
        assert in_shape[0] == -1, "First dimension of input should be batch"
        assert in_shape[1] == in_shape[2], "Expecting square input"
        assert num_transformers >= 1
        transfomers = [
            Transformer(inp=in_shape[3],
                        oup=dim_head,
                        image_size=in_shape[1:3],
                        heads=num_heads,
                        dim_head=dim_head,
                        dropout=dropout)
        ]
        for _ in range(num_transformers - 1):
            transfomers.append(
                Transformer(inp=dim_head,
                            oup=dim_head,
                            image_size=in_shape[1:3],
                            heads=num_heads,
                            dim_head=dim_head,
                            dropout=dropout))
        self.transformers = torch.nn.Sequential(*transfomers)
        self.gap = Reduce('b ih iw c  -> b c', 'mean')
        self.class_net = torch.nn.Linear(in_features=dim_head,
                                         out_features=10)

    def forward(self, x: torch.Tensor):
        x_ = self.transformers(x)
        x_ = self.gap(x_)
        return self.class_net(x_)
