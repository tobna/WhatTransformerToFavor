"""
Continue pretraining / finetuning after something went wrong.
"""

import logging
from data import prepare_dataset
from train import setup_model_optim_sched_scaler, setup_criteria_mixup, _train, setup_tracking_and_logging
from utils import prep_kwargs, ddp_setup, ddp_cleanup, save_model_state, load_pretrained
import torch
import os


def continue_training(model, **kwargs):
    model_path = model
    save_state = torch.load(model)

    # state is of the form
    #
    # state = {'epoch': epochs,
    #          'model_state': model.state_dict(),
    #          'optimizer_state': optimizer.state_dict(),
    #          'scheduler_state': scheduler.state_dict(),
    #          'args': dict(args),
    #          'run_name': run_name,
    #          'stats': metrics}

    args = prep_kwargs(save_state["args"])

    args.distributed, device, world_size, rank, gpu_id = ddp_setup()
    torch.cuda.set_device(device)

    if "world_size" in args and args.world_size is not None:
        global_bs = args.batch_size * args.world_size
    else:
        # assume global bs is given in kwargs
        global_bs = kwargs["batch_size"]
    args.batch_size = int(global_bs / world_size)
    args.world_size = world_size

    if "dataset" in args and args.dataset is not None:
        dataset = args.dataset
    else:
        # get default dataset for the task
        if args.task == "pre-train":
            dataset = "ImageNet21k"
        else:
            dataset = "ImageNet"
        args.dataset = dataset

    start_epoch = save_state["epoch"]
    if "epochs" in args and args.epochs is not None and args.epochs != start_epoch:
        epochs = args.epochs
    else:
        epochs = kwargs["epochs"]

    full_run_name, logging_file_name = setup_tracking_and_logging(args, rank, append_model_path=model_path)
    logging.info(f"Logging run {full_run_name} to file {logging_file_name}")

    # get the datasets & dataloaders
    train_loader, args.n_classes = prepare_dataset(dataset, args, rank=rank)
    val_loader, _ = prepare_dataset(dataset, args, train=False, rank=rank)

    model, args, _, __ = load_pretrained(model_path, args)

    model, optimizer, scheduler, scaler = setup_model_optim_sched_scaler(model, device, epochs, args)

    optimizer.load_state_dict(save_state["optimizer_state"])
    scheduler.load_state_dict(save_state["scheduler_state"])

    # log all devices
    logging.info(f"training on {device} -> {torch.cuda.get_device_name(device) if args.device != 'cpu' else ''}")
    if rank == 0:
        logging.info(f"torch version {torch.__version__}")
        logging.info(f"full set of arguments: {args}")

    if args.seed:
        torch.manual_seed(args.seed)

    criterion, val_criterion, mixup = setup_criteria_mixup(args)

    model_folder = f"{args.results_folder}/models/{full_run_name}/"
    os.makedirs(model_folder, exist_ok=True)
    if rank == 0:
        logging.info(f"start training at epoch {start_epoch}")
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
        scaler=scaler,
        do_metrics_calculation=True,
        start_epoch=start_epoch,
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
