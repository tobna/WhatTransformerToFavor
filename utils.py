"""
Utils and small helper functions
"""
import shutil
import warnings
import numpy as np
import torch
from math import ceil, log
from dataclasses import dataclass

from timm.utils import NativeScaler, dispatch_clip_grad
from torchvision.transforms import transforms

from config import *
import torch.distributed as dist
import os
from collections import abc
from math import cos, pi
import logging


@dataclass
class SchedulerArgs:
    """
    Class for scheduler arguments.

    Attributes
    ----------
    sched : str
        The type of learning rate scheduler to use.
    epochs : int
        The total number of epochs to train for.
    min_lr : float
        The minimum learning rate to use.
    warmup_lr : float
        The learning rate to use during the warmup phase.
    warmup_epochs : int
        The number of epochs to use for the warmup phase.
    cooldown_epochs : int
        The number of epochs to reduce the learning rate after training has completed.

    Examples
    --------
    >>> args = SchedulerArgs(sched='cosine', epochs=100, min_lr=0.0001, warmup_lr=0.001, warmup_epochs=5)
    """

    sched: str
    epochs: int
    min_lr: float
    warmup_lr: float
    warmup_epochs: int
    cooldown_epochs: int = 0


def scheduler_function_factory(epochs, sched, warmup_epochs=0, lr=None, min_lr=0., warmup_sched=None, warmup_lr=None, offset=-1, **kwargs):
    """Creates a scheduler factor function.

    Parameters
    ----------
    sched : str
        the learning rate schedule type
    epochs : int
        length of the full schedule
    warmup_epochs : int
        number of epochs reserved for warmup
    lr : float
        learning rate (has to be given, when warmup or min_lr are set)
    min_lr : float
        minimum learning rate
    warmup_sched : str
        the type of schedule during warmup
    warmup_lr : float
        (starting) learning rate during warmup

    Returns
    -------
    abc.Callable
        the scheduling factor function
    """
    sched = sched.lower()
    warmup_f = lambda ep: 1.
    if warmup_epochs > 0:
        assert warmup_lr is not None, f"Need warmup_lr, but got None"
        warmup_lr_factor = warmup_lr / lr
        if warmup_sched == 'linear':
            warmup_f = lambda ep: warmup_lr_factor + (1 - warmup_lr_factor) * max(ep, 0.) / warmup_epochs
        elif warmup_sched == 'const':
            warmup_f = lambda ep: warmup_lr_factor
        else:
            raise NotImplementedError(f"Warmup schedule {warmup_sched} not implemented")

    epochs = epochs - warmup_epochs + offset
    if sched == 'cosine':
        # cos from 0 to pi
        main_f = lambda ep: cos(pi * ep / epochs) / 2 + .5

    elif sched == 'const':
        main_f = lambda ep: 1.
    else:
        raise NotImplementedError(f"Schedule {sched} is not implemented.")

    # rescale and add min_lr
    min_lr_fact = min_lr / lr
    main_f_with_min_lr = lambda ep: (1 - min_lr_fact) * main_f(ep) + min_lr_fact

    return lambda ep: warmup_f(ep + offset) if ep + offset < warmup_epochs else main_f_with_min_lr(ep + offset - warmup_epochs)


class DotDict(dict):
    """
     Extension of a Python dictionary to access its keys using dot notation.

    Parameters
    ----------
    dict : dict
        The dictionary to be extended.

    Example
    -------
    Create a DotDict object and access its keys using dot notation.

    >>> my_dict = {"key1": "value1", "key2": 2, "key3": True}
    >>> my_dot_dict = DotDict(my_dict)
    >>> my_dot_dict.key1
    'value1'
    >>> my_dot_dict.key2
    2
    >>> my_dot_dict.key3
    True
    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item, default=None):
        if item not in self:
            return default
        return self.get(item)


def prep_kwargs(kwargs):
    """Prepare the arguments and add defaults.

    Parameters
    ----------
    kwargs : dict[str, Any]
        dict of kwargs

    Returns
    -------
    DotDict
        prepared args
    """

    for k, v in default_kwargs.items():
        if k not in kwargs:
            kwargs[k] = v

    for var_name in ["dataset_root", "results_folder", "logging_folder"]:
        if var_name not in kwargs:
            kwargs[var_name] = globals()[var_name]

    return DotDict(kwargs)


__pos_encoding_matrices = {'trig': {}}


def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    operation = transforms.Normalize(mean=[-mu/sigma for mu, sigma in zip(mean, std)], std=[1/sigma for sigma in std])
    return operation(x)


def ddp_setup():
    """Set up the distributed environment.

    Returns
    -------
    tuple
        A tuple containing the following elements:
            * bool: Whether the training is distributed.
            * torch.device: The device to use for distributed training.
            * int: The total number of processes in the distributed setup.
            * int: The global rank of the current process in the distributed setup.
            * int: The local rank of the current process on its node.

    Notes
    -----
    The 'nccl' backend is used.
    """

    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    distributed = 'RANK' in os.environ
    if distributed:
        # print(f"rank {rank}, local rank {local_rank}, world size {num_gpus}")
        # print(f"cuda available {torch.cuda.is_available()} on {torch.cuda.device_count()} devices")
        dist.init_process_group('nccl')
    return distributed, torch.device(f'cuda:{local_rank}'), num_gpus, rank, local_rank


def ddp_cleanup():
    """
    Cleans the distributed setup after use.
    """
    if 'RANK' in os.environ:
        dist.destroy_process_group()


def set_filter_warnings():
    """
    Filter out some warnings to reduce spam
    """
    # filter DataLoader number of workers warning
    warnings.filterwarnings("ignore",
                            ".*worker processes in total. Our suggested max number of worker in current system is.*")

    # Filter datadings only varargs warning
    warnings.filterwarnings("ignore", ".*only accepts varargs so.*")

    # Filter warnings from calculation of MACs & FLOPs
    # warnings.filterwarnings("ignore", ".*No handlers found:.*")

    # Filter warnings from gather
    warnings.filterwarnings("ignore", ".*is_namedtuple is deprecated, please use the python checks instead.*")

    # Filter warnings from meshgrid
    warnings.filterwarnings("ignore", ".*in an upcoming release, it will be required to pass the indexing.*")


def remove_prefix(state_dict, prefix="module."):
    """
    Remove a prefix from the keys in a state dictionary.

    Parameters
    ----------
    state_dict : dict[str, Any]
        The state dictionary to remove the prefix from.
    prefix : str, optional
        The prefix to remove from the keys. Default is 'module.'.

    Returns
    -------
    dict[str, Any]
        A new dictionary with the prefix removed from the keys.

    Examples
    --------
    >>> state_dict = {'module.layer1.weight': 1, 'module.layer1.bias': 2}
    >>> remove_prefix(state_dict)
    {'layer1.weight': 1, 'layer1.bias': 2}
    """

    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def prime_factors(n):
    """
    Calculate the prime factors of a given integer.

    Parameters
    ----------
    n : int
        The integer to find the prime factors of.

    Returns
    -------
    list[int]
        A list of integers representing the prime factors of the input integer.
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def linear_regession(points):
    """Calculate a linear interpolation of the points

    Parameters
    ----------
    points : dict[float, float]
        points to interpolate in the format points[x] = y

    Returns
    -------
    abc.Callable
        interpolation function x |-> y
    """

    N = len(points)
    x = []
    y = []
    for x_i, y_i in points.items():
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    y = np.array(y)

    a = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x * x).sum() - x.sum() ** 2)
    b = (y.sum() - a * x.sum()) / N
    return lambda z: a * z + b


def save_model_state(model_folder, epoch, args, model_state, regular_save=True, stats=None, val_accs=None, epoch_accs=None, additional_reason='', **kwargs):
    """

    Parameters
    ----------
    model_folder : str
        folder path to save model in
    epoch : int
        the current epoch
    args : dict[str, Any]
        all arguments
    model_state : dict[str, torch.Tensor]
        model state dict
    stats : dict[str, float]
        current training statistics
    val_accs : dict[str, float]
        validation accuracies, if not given in stats
    epoch_accs : dict[str, float]
        training accuracies, if not given in stats
    additional_reason : str
        reason model is saved, other than creating a generic checkpoint; this will overwrite the file name
    kwargs
        further arguments to be saved, give 'optimizer_state' and 'scheduler_state'

    Returns
    -------

    """
    # make args dict, not DotDict to be able to save it
    state = {'epoch': epoch, 'model_state': model_state, 'run_name': args.run_name, 'args': dict(args)}
    if stats is None:
        stats = {}
    if val_accs is not None:
        stats = {**stats, **val_accs}
    if epoch_accs is not None:
        stats = {**stats, **epoch_accs}
    state['stats'] = stats
    state = {**state, **kwargs}
    logging.info(f"saving model state at epoch {epoch} ({additional_reason})")
    regular_file_name = f"ep_{epoch}{f'_acc_{int(100 * min(val_accs.values()))}' if val_accs is not None else ''}.tar"
    if len(additional_reason) > 0:
        save_name = additional_reason + ".tar"
    else:
        save_name = regular_file_name
    outfile = os.path.join(model_folder, save_name)
    torch.save(state, outfile)
    if len(additional_reason) > 0 and regular_save:
        shutil.copyfile(outfile, os.path.join(model_folder, regular_file_name))


class ScalerGradNormReturn(NativeScaler):
    """
    A wrapper around PyTorch's NativeScaler that returns the gradient norm.

    Notes
    -----
    This class is a wrapper around PyTorch's NativeScaler that returns the gradient norm of a subset of the model's
    parameters after scaling and backpropagation. The selected parameters are determined by the `parameters` argument.
    If `parameters` is None, the gradient norm is not computed.

    Examples
    --------
    >>> scaler = ScalerGradNormReturn()
    >>> loss = compute_loss(...)
    >>> optimizer.zero_grad()
    >>> grad_norm = scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters())
    """
    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        """
        Scale and backpropagate through the loss tensor, and return the gradient norm of the selected parameters.
        Does an optimizer step.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor to scale and backpropagate through.
        optimizer : torch.optim.Optimizer
            The optimizer to use for the optimization step.
        clip_grad : float or None, optional (default=None)
            The maximum allowed norm of the gradients. If None, no clipping is performed.
        clip_mode : str, optional (default='norm')
            The mode used for clipping the gradients. Only used if `clip_grad` is not None. Possible values are 'norm'
            (clipping the norm of the gradients) and 'value' (clipping the value of the gradients).
        parameters : iterable of torch.nn.Parameter or None, optional (default=None)
            The parameters to compute the gradient norm for. If None, the gradient norm is not computed.
        create_graph : bool, optional (default=False)
            Whether to create a computation graph for computing second-order gradients.

        Returns
        -------
        float
            The gradient norm of the selected parameters.
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)

        # always unscale the gradients, since it's being done anyway
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        if parameters is not None:
            grads = [p.grad for p in parameters if p.grad is not None]
            device = grads[0].device
            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
        else:
            grad_norm = -1
        if clip_grad is not None:
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()
        return grad_norm


class NoScaler:
    """
    Dummy gradient scaler that doesn't scale gradients.

    This scaler performs a simple backward pass with the given loss, and then updates the model's parameters
    with the given optimizer. The resulting gradient norm is computed and returned.
    """
    def __call__(self, loss, optimizer, parameters=None, **kwargs):
        """
        Performs backward pass with the given loss, updates the model's parameters with the given optimizer, and
        computes the resulting gradient norm.

        Parameters
        ----------
        loss : torch.Tensor
            The loss tensor that the gradients will be computed from.
        optimizer : torch.optim.Optimizer
            The optimizer that will be used to update the model's parameters.
        parameters : iterable of torch.Tensor, optional (default=None)
            An iterable of model parameters to compute gradients. If None, returns -1.
        **kwargs
            Additional keyword arguments; nothing will be done with these.

        Returns
        -------
        float
            The gradient norm computed after the optimizer step, if parameters is not None.
        """
        loss.backward()
        if parameters is not None:
            grads = [p.grad for p in parameters if p.grad is not None]
            device = grads[0].device
            grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2)
        else:
            grad_norm = -1
        optimizer.step()
        return grad_norm
