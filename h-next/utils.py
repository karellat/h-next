import ast
import importlib
import json
import os
from typing import List, Any, Tuple

import click
import matplotlib.pyplot as plt
import sklearn
import torch.nn
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable


def retype_str(text: str):
    # NOTE: https://stackoverflow.com/questions/55921050/python-getting-scikit-learn-classifier-by-passing-string-with-the-model-name-a
    if text.isdigit():
        return int(text)
    elif text.replace('.', '', 1).isdigit() and text.count('.') < 2:
        return float(text)
    else:
        return text


def cbar_plot(fig, ax, img, cmap='hot'):
    ax.set_axis_off()
    im = ax.imshow(img, interpolation='None', cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


def make_grid(imgs, nrow, ncol, title="", figsize=None, cmap='gray', labels=None, cbar=True) -> plt.Figure:
    if figsize is None:
        figsize = (2*ncol, 2*nrow)
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    if labels is not None:
        assert len(imgs) == len(labels)
    for idx, img in enumerate(imgs):
        ax = fig.add_subplot(nrow, ncol, idx + 1)
        ax.set_axis_off()
        im = ax.imshow(img, interpolation='None', cmap=cmap)
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        if labels is not None:
            ax.set_title(str(labels[idx]))
    return fig


def get_model_by_name(
        # NOTE: https://stackoverflow.com/questions/55921050/python-getting-scikit-learn-classifier-by-passing-string-with-the-model-name-a
        model_name: str, import_module: str, model_params: dict
) -> sklearn.base.BaseEstimator:
    # Parse model params
    if type(model_params) is str:
        if len(model_params) > 0:
            model_params = {arg.split('=')[0]: retype_str(arg.split('=')[1]) for arg in model_params.split(',')}
        else:
            model_params = dict()

    """Returns a scikit-learn or xgboost model."""
    model_class = getattr(importlib.import_module(import_module), model_name)
    model = model_class(**model_params)  # Instantiates the model
    return model



def log_wandb(wandb_key_json: str = "wandb_key.json"):
    assert os.path.exists(wandb_key_json), f"Wandb json not found at. {wandb_key_json}"

    # Opening JSON file
    with open(wandb_key_json, 'r') as f:
        logger.debug(f"Loading Wandb key from {wandb_key_json}")
        os.environ["WANDB_API_KEY"] = json.load(f)["key"]


class ClickDictionaryType(click.ParamType):
    name = "dict"

    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return value

        try:
            return ast.literal_eval(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid dictionary", param, ctx)


def get_all_module_of_type(module: torch.nn.Module, sample_type: Any) -> List[Tuple[str, torch.nn.Module]]:
    all_children = []
    for name, child in module.named_children():
        if type(child) is sample_type:
            all_children.append((name, child))
        all_children += get_all_module_of_type(child, sample_type)
    return all_children
