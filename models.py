"""
Model loading and preparation
"""
from functools import partial

import timm
import torch.nn as nn
import torch
import utils
from resizing_interface import vit_sizes
from architectures.vit import TimmViT
import importlib
import os

# import all architectures
for architecture in os.listdir('architectures'):
    if not architecture.endswith('.py'):
        continue
    try:
        importlib.import_module(f'architectures.{architecture[:-3]}')
    except:
        print(f'\033[93mCould not import \033[0m\033[91m{architecture}\033[0m')


def prepare_model(model_str, args):
    """Prepare a new model.
    If the name is of the format ViT-<size>/<patch_size>, use a *TimmViT*, else fall back to timm model loading.

    Parameters
    ----------
    model_str : str
        model name
    args : utils.DotDict
        further arguments, needs to have keys n_classes, drop_path_rate;
        key imsize or '_<imsize>' at the end of ViT specification

    Returns
    -------
    nn.Module
        the new model
    """

    kwargs = dict(args)
    if args.layer_scale_init_values:
        kwargs["init_values"] = kwargs["init_scale"] = args.layer_scale_init_values
    if args.dropout and args.dropout > 0.:
        kwargs["drop"] = kwargs["drop_rate"] = args.dropout
    if args.drop_path_rate and args.drop_path_rate > 0.:
        kwargs["drop_block_rate"] = args.drop_path_rate
    kwargs["num_classes"] = args.n_classes
    kwargs["img_size"] = args.imsize
    if model_str.startswith("ViT"):
        # Format: ViT-{Ti,S,B,L}/<patch_size>[_<image_res>]
        h1, h2 = model_str.split("/")
        _, model_size = h1.split("-")
        if "_" in h2:
            patch_size, image_res = h2.split("_")
            assert args.imsize is None or args.imsize == int(image_res), \
                f"Got two different image sizes: {args.imsize} vs {image_res}"
        else:
            patch_size = h2
        model = TimmViT(patch_size=int(patch_size), in_chans=3, **vit_sizes[model_size],
                       norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    else:
        model = timm.create_model(model_str, pretrained=False, **kwargs)
    return model


def load_pretrained(model_path, args, new_dataset_params=False):
    """Load a pretrained model from .tar file.

    Parameters
    ----------
    new_dataset_params : bool
        change model parameters (imsize, n_classes) to the ones from args.
    model_path : str
        path to .tar file
    args : Any
        new model parameters

    Returns
    -------
    tuple[nn.Module, dict, dict, dict]
        loaded model, args, old_args, save_state
    """
    save_state = torch.load(model_path, map_location='cpu')
    old_args = utils.prep_kwargs(save_state["args"])
    args.model = old_args.model

    if old_args.model.startswith('flash_vit'):
        args.pop("layer_scale_init_values", None)
        old_args.pop("layer_scale_init_values", None)

    # load the model (the old one first)
    model = prepare_model(old_args.model, old_args)
    file_save_state = utils.remove_prefix(save_state["model_state"], prefix="_orig_mod.")
    file_save_state = utils.remove_prefix(file_save_state)
    try:
        model.load_state_dict(file_save_state)
    except (UnboundLocalError, RuntimeError) as e:
        model_keys = set(model.state_dict().keys())
        file_keys = set(file_save_state.keys())
        print(f"Error loading state dict: {e}")
        model_minus_file = model_keys.difference(file_keys)
        file_minus_model = file_keys.difference(model_keys)
        print(f"model-file: {model_minus_file}\nfile-model: {file_minus_model}")
        if len(file_minus_model) == 0 and all(['.ls' in key and key.endswith('.gamma') for key in model_minus_file]):
            print(f"Old model was without LayerScale -> replicating")
            try:
                args.pop("layer_scale_init_values")
                old_args.pop("layer_scale_init_values")
                model = prepare_model(old_args.model, old_args)
                model.load_state_dict(file_save_state)
            except (UnboundLocalError, RuntimeError) as e:
                print(f"Could not resolve conflict")
                print(f"Still got error {e}")
                exit(-1)
        else:
            exit(-1)

    if new_dataset_params:
        # setup for finetuning parameters
        model.set_image_res(args.imsize)
        model.set_num_classes(args.n_classes)

    return model, args, old_args, save_state
