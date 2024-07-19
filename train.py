import sys
from contextlib import nullcontext
from os.path import isfile

import os
import numpy as np
import torch
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from time import time

from utils import (
    prep_kwargs,
    ddp_setup,
    ddp_cleanup,
    save_model_state,
    set_filter_warnings,
    SchedulerArgs,
    ScalerGradNormReturn,
    NoScaler,
)
import torch.optim as optim
from tqdm import tqdm
import random
from math import isfinite
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from data import prepare_dataset
from models import load_pretrained, prepare_model
from metrics import accuracy, calculate_metrics

try:
    from apex.optimizers import FusedLAMB  # noqa: F401

    apex_available = True
except ImportError:
    print("Nvidia apex not available")
    apex_available = False
try:
    from lion_pytorch import Lion

    lion_available = True
except ImportError:
    print("Lion not available")
    lion_available = False


def finetune(model, dataset, epochs, **kwargs):
    """Finetune a pretrained model on a given dataset for a specified number of epochs.

    Parameters
    ----------
    model : str
        Path to the pretrained model state file (in .tar format).
    dataset : str
        Name of the dataset to finetune on.
    epochs : int
        Number of epochs to train for.
    **kwargs : dict
        Further arguments for model setup, training, evaluation,...

    Returns
    -------
    None

    Notes
    -----
    This function assumes that the model was pretrained on a different dataset using a different set of hyperparameters.
    It fine-tunes the model on a new dataset by loading the pretrained weights and training for the specified number of
    epochs. The function supports distributed training using the PyTorch DistributedDataParallel module.
    """

    set_filter_warnings()

    # Add defaults & make keys properties
    args = prep_kwargs(kwargs)

    args.dataset = dataset
    args.epochs = epochs

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    args.world_size = world_size
    torch.cuda.set_device(device)
    args.batch_size = int(args.batch_size / world_size)

    if args.seed is not None:
        # fix the seed for reproducibility
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # get the datasets & dataloaders
    # transform only contains resize & crop here; everything else is handled on the GPU / in the training loop
    train_loader, args.n_classes = prepare_dataset(dataset, args, rank=rank)
    val_loader, _ = prepare_dataset(dataset, args, train=False, rank=rank)

    model, args, old_args, save_state = load_pretrained(model, args, new_dataset_params=True)

    full_run_name, logging_file_name = setup_tracking_and_logging(args, rank)

    if rank == 0:
        logging.info(f"The model was pretrained on {old_args.dataset} for {save_state['epoch']} epochs.")

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)

    # log all devices
    logging.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logging.info(f"torch version {torch.__version__}")
        logging.info(f"full set of arguments: {args}")
        logging.info(f"full set of old arguments: {old_args}")

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    model_folder = f"{args.results_folder}/models/{full_run_name}/"
    os.makedirs(model_folder, exist_ok=True)
    if rank == 0:
        logging.info(f"Run name: '{full_run_name}'")
        logging.info(f"Logging file name: '{logging_file_name}'")

    metrics = _train(
        model,
        train_loader,
        optimizer,
        rank,
        epochs,
        device,
        mixup,
        criterion,
        world_size,
        scheduler,
        args,
        val_loader,
        val_criterion,
        model_folder,
        scaler,
        do_metrics_calculation=True,
        show_tqdm=args.tqdm,
    )

    ddp_cleanup()
    if rank == 0:
        logging.info(f"Run '{full_run_name}' is done.")
        save_model_state(
            model_folder,
            epochs,
            args,
            model_state=model.state_dict(),
            stats=metrics,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            additional_reason="final",
            regular_save=False,
        )


def pretrain(model, dataset, epochs, **kwargs):
    """
    Train or pretrain a model.

    Parameters
    ----------
    model : str
        Name of the model to train.
    dataset : str
        Name of the dataset to train the model on.
    epochs : int
        Number of training epochs.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    None

    Notes
    -----
    This function sets up logging, prepares the model, and trains the model on the given dataset.

    """

    set_filter_warnings()

    # Add defaults & make args properties
    args = prep_kwargs(kwargs)
    args.dataset = dataset
    args.model = model
    args.epochs = epochs

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    args.world_size = world_size
    torch.cuda.set_device(device)
    args.batch_size = int(args.batch_size / world_size)

    if args.seed is not None:
        # fix the seed for reproducibility
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    full_run_name, logging_file_name = setup_tracking_and_logging(args, rank)

    # get the datasets & dataloaders
    train_loader, args.n_classes = prepare_dataset(dataset, args, rank=rank)
    val_loader, _ = prepare_dataset(dataset, args, train=False, rank=rank)

    # setup model with amp & DDP (continued after logging setup)
    if isinstance(model, str):
        if model.startswith("ViT") and "_" not in model:
            model += f"_{args.imsize}"
        model_name = model
        model = prepare_model(model, args)
    if not model_name:
        model_name = type(model).__name__

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)

    # log all devices
    logging.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logging.info(f"python version {sys.version}")
        logging.info(f"torch version {torch.__version__}")
        logging.info(f"full set of arguments: {args}")

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    model_folder = f"{args.results_folder}/models/{full_run_name}/"
    os.makedirs(model_folder, exist_ok=True)
    if rank == 0:
        logging.info(f"Run name: '{full_run_name}'")
        logging.info(f"Logging file name: '{logging_file_name}'")

    metrics = _train(
        model,
        train_loader,
        optimizer,
        rank,
        epochs,
        device,
        mixup,
        criterion,
        world_size,
        scheduler,
        args,
        val_loader,
        val_criterion,
        model_folder,
        scaler,
        do_metrics_calculation=True,
        show_tqdm=args.tqdm,
    )

    ddp_cleanup()
    if rank == 0:
        logging.info(f"Run '{full_run_name}' is done.")
        save_model_state(
            model_folder,
            epochs,
            args,
            model_state=model.state_dict(),
            stats=metrics,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            additional_reason="final",
        )


def setup_tracking_and_logging(args, rank, append_model_path=None):
    """
    Sets up logging and tracking for an experiment.

    Parameters
    ----------
    args : Namespace
        Parsed command-line arguments
    rank : int
        The rank of the current process
    append_model_path : str, optional
        Path of an existing model, by default None

    Returns
    -------
    Tuple[str, str, str]
        A tuple containing the full run name, the logging file name, and the experiment ID.

    Raises
    ------
    AssertionError
        If `dataset` or `model` is `None`.
        If `'%'` is found in `args.run_name`.

    Notes
    -----
    This function sets up logging to stdout and file, as well as MLflow tracking for an experiment.

    """

    dataset, model, epochs = args.dataset, args.model, args.epochs
    assert dataset is not None and model is not None

    full_run_name = None
    if append_model_path is not None:
        if isfile(args.logging_folder + "/" + append_model_path.split("/")[-2] + ".log"):
            full_run_name = append_model_path.split("/")[-2]
        elif isfile(
            args.logging_folder
            + "/"
            + append_model_path.split("/")[-3]
            + "_"
            + append_model_path.split("/")[-2]
            + ".log"
        ):
            full_run_name = append_model_path.split("/")[-3] + "/" + append_model_path.split("/")[-2]
        args.run_name = full_run_name.split("_")[1]

    assert "%" not in args.run_name, f"found '%' in run_name '{args.run_name}'. This messes with string formatting..."

    # logging to stdout & file
    if full_run_name is None:
        full_run_name = (
            f"{args.task.replace('-', '')}_{args.run_name}_{model}_{dataset}_"
            f"{datetime.now().strftime('%d.%m.%Y_%H:%M:%S')}"
        )
    logging_file_name = f"{full_run_name}.log".replace("/", "_")
    logging.basicConfig(
        format=f"%(asctime)s [{args.run_name}/%(threadName)s -> {rank}] %(levelname)s: %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(args.logging_folder + "/" + logging_file_name), logging.StreamHandler()],
    )

    if rank == 0:
        logging.info(f"{args.task.replace('-', '').capitalize()} {model} on {dataset} for {epochs} epochs")

    return full_run_name, logging_file_name


def setup_model_optim_sched_scaler(model, device, epochs, args):
    """Setup model, optimizer, and scheduler with automatic mixed precision (amp) and distributed data parallel (DDP).

    Parameters
    ----------
    model : nn.Module
        the loaded model
    device : torch.device
        the current device
    epochs : int
        total number of epochs to learn for (for scheduler)
    args
        further arguments

    Returns
    -------
    tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR]
        model, optimizer, & lr scheduler
    """
    model = model.to(device)

    if args.opt == "lion" and not lion_available:
        args.opt = "fusedlamb"
        logging.warning("Falling back from lion to fusedlamb")
    if args.opt == "fusedlamb" and not apex_available:
        args.opt = "adamw"
        logging.warning("Falling back from fusedlamb to adamw")
    if args.opt == "lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = create_optimizer(args, model)

    scaler = ScalerGradNormReturn() if args.amp else None

    # if args.model_ema:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     ema_model = ModelEma(model, decay=args.model_ema_decay, resume='')

    if args.distributed:
        model = DDP(model, device_ids=[device])

    if args.compile_model:
        try:
            model = torch.compile(model)
        except:  # noqa: E722
            logging.info("Could not compile the model.")

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer,
    #                                         lr_lambda=scheduler_function_factory(**args))
    sched_args = SchedulerArgs(args.sched, args.epochs, args.min_lr, args.warmup_lr, args.warmup_epochs)
    scheduler, _ = create_scheduler(sched_args, optimizer)

    return model, optimizer, scheduler, scaler


def setup_criteria_mixup(args):
    """Setup further objects that are needed for training.

    Parameters
    ----------
    args
        arguments

    Returns
    -------
    tuple[nn.Module, nn.Module, nn.Module | None, Mixup]
        criterion, validation criterion, & mixup
    """
    if args.aug_cutmix:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    val_criterion = LabelSmoothingCrossEntropy(smoothing=0.0)

    mixup = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, label_smoothing=args.label_smoothing, num_classes=args.n_classes)
    return criterion, val_criterion, mixup


def _train(
    model,
    train_loader,
    optimizer,
    rank,
    epochs,
    device,
    mixup,
    criterion,
    world_size,
    scheduler,
    args,
    val_loader,
    val_criterion,
    model_folder,
    scaler,
    do_metrics_calculation=True,
    start_epoch=0,
    show_tqdm=True,
    topk=(1, 5),
    acc_dict_key="acc{}",
):

    training_start = time()
    time_spend_training = time_spend_validating = 0
    current_best_acc = 0.0
    if rank == 0:
        logging.info(f"Dataloader has {len(train_loader)} batches")

    for epoch in range(start_epoch, epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        epoch_time, epoch_loss_rel, epoch_accs = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            rank,
            epoch,
            device,
            mixup,
            criterion,
            world_size,
            scheduler,
            scaler,
            args,
            topk,
            acc_dict_key,
            show_tqdm,
        )
        time_spend_training += epoch_time

        val_time, val_loss_rel, val_accs = _evaluate(
            model, val_loader, epoch, rank, device, world_size, val_criterion, args, topk, acc_dict_key
        )
        time_spend_validating += val_time

        if rank == 0:
            logging.info(f"total_time={time() - training_start}s")

        # saving current state
        if rank == 0 and (min(val_accs.values()) > current_best_acc or (epoch + 1) % args.save_epochs == 0):
            reason = "top" if min(val_accs.values()) > current_best_acc else ""  # min(...) will be the top-1 accuracy
            if reason == "top":
                current_best_acc = min(val_accs.values())
                logging.info(f"found a new best model with accuracy: {current_best_acc} ({val_accs})")
            kwargs = dict(
                model_state=model.state_dict(),
                stats={"train_loss": epoch_loss_rel, "val_loss": val_loss_rel},
                val_accs=val_accs,
                epoch_accs=epoch_accs,
                optimizer_state=optimizer.state_dict(),
                additional_reason=reason,
                regular_save=(epoch + 1) % args.save_epochs == 0,
            )
            if scheduler:
                kwargs["scheduler_state"] = scheduler.state_dict()
            save_model_state(model_folder, epoch + 1, args, **kwargs)

    if rank == 0:
        end_time = time()
        logging.info(
            f"training done: total time={end_time - training_start}, "
            f"time spend training={time_spend_training}, "
            f"time spend validating={time_spend_validating}"
        )

    results = {"train_loss": epoch_loss_rel, "val_loss": val_loss_rel, **val_accs, **epoch_accs}
    if do_metrics_calculation:
        if rank == 0:
            # Calculate efficiency metrics
            inp = next(iter(train_loader))[0].to(device)
        else:
            inp = None
        metrics = calculate_metrics(
            args, model, rank=rank, input=inp, device=device, did_training=True, world_size=world_size
        )

        if rank == 0:
            logging.info(f"Efficiency metrics: {metrics}")
    return results


def _evaluate(
    model, val_loader, epoch, rank, device, world_size, val_criterion, args, topk=(1, 5), acc_dict_key="acc{}"
):
    """Evaluate the model

    Parameters
    ----------
    model : nn.Module
        the model to evaluate
    val_loader : DataLoader
        loader for evaluation data
    epoch : int
        the current epoch (for logging & tracking)
    rank : int
        this processes rank (don't log n times)
    device : torch.device
        device to evaluate on
    world_size : int
        number of processes / GPUs
    val_criterion : nn.Module
        validation loss

    Returns
    -------
    tuple[float, float, float]
        validation time [s], average validation loss, validation accuracy
    """

    acc_dict_key = "val_" + acc_dict_key

    model.eval()
    val_loss = 0
    val_accs = {acc_dict_key.format(k): 0.0 for k in topk}
    val_start = time()
    n_iters = 0
    iterator = (
        tqdm(val_loader, total=len(val_loader), desc=f"Validating epoch {epoch + 1}")
        if rank == 0 and args.tqdm
        else val_loader
    )
    for xs, ys in iterator:
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast() if args.eval_amp else nullcontext():
                preds = model(xs)

        val_loss += val_criterion(preds, ys).item()
        for key, val in accuracy(preds, ys, topk=topk, dict_key=acc_dict_key).items():
            val_accs[key] += val
        n_iters += 1

    if args.distributed:
        dist.barrier()
    val_end = time()
    iterations = n_iters
    val_accs = {key: val / iterations for key, val in val_accs.items()}
    val_loss = val_loss / iterations

    if args.distributed:
        gather_tensor = torch.Tensor([val_loss, *[val_accs[acc_dict_key.format(k)] for k in topk]]).to(device)
        dist.barrier()
        dist.all_reduce(gather_tensor)
        gather_tensor = (gather_tensor / world_size).tolist()
        val_loss = gather_tensor[0]
        for i, k in enumerate(topk):
            val_accs[acc_dict_key.format(k)] = gather_tensor[i + 1]

    if rank == 0:
        log_s = f"epoch {epoch + 1}: validation_loss={val_loss:.5f}, validation_time={val_end - val_start}s"
        for key, val in val_accs.items():
            log_s += f", {key}={val * 100}%"
        logging.info(log_s)
    return val_end - val_start, val_loss, val_accs


def _train_one_epoch(
    model,
    train_loader,
    optimizer,
    rank,
    epoch,
    device,
    mixup,
    criterion,
    world_size,
    scheduler,
    scaler,
    args,
    topk=(1, 5),
    acc_dict_key="acc{}",
    show_tqdm=True,
):
    model.train()
    iterator = (
        tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epoch + 1}")
        if rank == 0 and show_tqdm
        else train_loader
    )

    if not args.amp:
        scaler = NoScaler()

    epoch_loss = 0
    epoch_accs = {acc_dict_key.format(k): 0.0 for k in topk}
    epoch_start = time()
    grad_norms = []
    n_iters = 0
    for xs, ys in iterator:
        optimizer.zero_grad()
        n_iters += 1
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        if args.aug_cutmix:
            xs, ys = mixup(xs, ys)

        with torch.cuda.amp.autocast() if args.amp else nullcontext():
            preds = model(xs)
            loss = criterion(preds, ys) + (
                model.get_internal_loss() if hasattr(model, "get_internal_loss") else model.module.get_internal_loss()
            )

        if not isfinite(loss.item()):
            logging.error(f"Got loss value {loss.item()}. Stopping training.")
            logging.info(f"input has nan: {xs.isnan().any().item()}")
            logging.info(f"target has nan: {ys.isnan().any().item()}")
            logging.info(f"output has nan: {preds.isnan().any().item()}")
            for name, param in model.named_parameters():
                if param.isnan().any().item():
                    logging.info(f"parameter {name} has a nan value")
            if len(grad_norms) > 0:
                grad_norms = torch.Tensor(grad_norms)
                logging.info(
                    f"Gradient norms until now: min={grad_norms.min().item()}, 20th %tile={torch.quantile(grad_norms, .2).item()}, mean={torch.mean(grad_norms)}, 80th %tile={torch.quantile(grad_norms, .8).item()}, max={grad_norms.max()}"
                )
            sys.exit(1)

        iter_grad_norm = scaler(loss, optimizer, parameters=model.parameters())

        if args.gather_stats_during_training:
            if isfinite(iter_grad_norm):
                grad_norms.append(iter_grad_norm)

            if args.aug_cutmix:
                ys = ys.argmax(dim=-1)  # for accuracy with CutMix, just use the argmax for both

            epoch_loss += loss.item()
            accuracies = accuracy(preds, ys, topk=topk)
            for key in accuracies.keys():
                epoch_accs[key] += accuracies[key]

    if args.distributed:
        dist.barrier()
    epoch_end = time()

    iterations = n_iters
    epoch_accs = {key: val / iterations for key, val in epoch_accs.items()}
    epoch_loss = epoch_loss / iterations
    grad_norm_avrg = -1
    inf_grads = iterations - len(grad_norms)
    if len(grad_norms) > 0 and args.gather_stats_during_training:
        grad_norm_max = max(grad_norms)
        grad_norms = torch.Tensor(grad_norms)
        grad_norm_20 = torch.quantile(grad_norms, 0.2).item()
        grad_norm_80 = torch.quantile(grad_norms, 0.8).item()
        grad_norm_avrg = torch.mean(grad_norms)

    if args.distributed:
        # grad norm is already synchronized
        gather_tensor = torch.Tensor([epoch_loss, *[epoch_accs[acc_dict_key.format(k)] for k in topk]]).to(device)
        dist.barrier()
        dist.all_reduce(gather_tensor)
        gather_tensor = (gather_tensor / world_size).tolist()
        epoch_loss = gather_tensor[0]
        for i, k in enumerate(topk):
            epoch_accs[acc_dict_key.format(k)] = gather_tensor[i + 1]

    if rank == 0:
        lr = optimizer.param_groups[0]["lr"]
        if args.gather_stats_during_training:
            print_s = (
                f"epoch {epoch + 1}: loss={epoch_loss:.5f}, time={epoch_end - epoch_start}s, " f"learning rate={lr}"
            )
            for key, val in epoch_accs.items():
                print_s += f", {key}={val*100}%"
            logging.info(print_s)
            if len(grad_norms) > 0:
                logging.info(
                    f"epoch {epoch + 1}: grad norm avrg={grad_norm_avrg}, grad norm max={grad_norm_max}, "
                    f"inf grad norm={inf_grads}, grad norm 20%={grad_norm_20}, grad norm 80%={grad_norm_80}"
                )
            else:
                logging.info(f"epoch {epoch + 1}: inf grad norm={inf_grads}")
                logging.warning("100% of update steps with infinite grad norms!")
        else:
            logging.info(f"epoch {epoch + 1}: time={epoch_end - epoch_start}s, " f"learning rate={lr}")

    if scheduler:
        if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
            scheduler.step()
        else:
            scheduler.step(epoch)

    if args.gather_stats_during_training:
        return epoch_end - epoch_start, epoch_loss, epoch_accs
    return epoch_end - epoch_start, -1, {}
