#!/usr/bin/env python3

"""
Parse args and call the correct script inside slurm container.
Outside the container, on the head-node, create and call the correct srun command.
"""

import argparse
from config import *
import os


def base_parser():
    """Creates the argument parser with all the choices for the training / evaluation scripts

    Returns
    -------
    argparse.ArgumentParser
        parser
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Main
    group = parser.add_argument_group("Main")
    group.add_argument("-task", nargs='?', choices=["pre-train", "fine-tune", "eval", "parser-test", "eval-metrics",
                                                    "continue"], default="pre-train", help="Task to perform.")
    group.add_argument("-model", nargs='?', type=str, required=True,
                       help="Model to use. Either model name for a new model or weights "
                            "and dicts to load for fine-tuning.")
    group.add_argument("-dataset", nargs='?', type=str, help="Dataset to train on. (default depends on the task)")
    group.add_argument("-epochs", nargs='?', type=int, help="Number of epochs to train.")
    group.add_argument("-run_name", nargs='?', type=str,
                       help="A name for the run. If not give, the model name is used instead.")

    # Further model parameters
    group = parser.add_argument_group("Further model parameters")
    group.add_argument("-drop_path_rate", nargs='?', type=float, default=default_kwargs["drop_path_rate"],
                       help="Drop path rate for ViT models.")
    group.add_argument("-layer_scale_init_values", nargs='?', type=float,
                       default=default_kwargs["layer_scale_init_values"], help="LayerScale initial values.")
    group.add_argument("-no_layer_scale", action="store_true", help="Don't use layer scale.")
    group.add_argument("-no_qkv_bias", action="store_true",
                       help="Don't use bias in linear transformation to queries, keys, and values.")
    group.add_argument("-pre_norm", action='store_true', help="Use norm first architecture.")
    group.add_argument("-dropout", nargs='?', type=float, default=default_kwargs["dropout"], help="Model dropout.")
    # group.add_argument("-no_model_ema", action="store_true",
    #                    help="Don't use an exponential moving average for model parameters")
    # group.add_argument("-model_ema_decay", nargs='?', type=float, default=default_kwargs["model_ema_decay"],
    #                    help="Decay rate for exponential moving average of model parameters")

    # Experiment management
    group = parser.add_argument_group("Experiment management")
    group.add_argument("-seed", nargs='?', default=default_kwargs['seed'], type=int, help="Manual RNG seed.")
    group.add_argument("-experiment_name", nargs='?', default=default_kwargs["experiment_name"], type=str,
                       help="Name for the experiment as a prefix in ML-Flow.")
    group.add_argument("-save_epochs", nargs='?', default=default_kwargs["save_epochs"], type=int,
                       help="Number of epochs after which to save the full training state.")
    group.add_argument("-dataset_root", nargs='?', default=dataset_root, type=str,
                       help="Root folder for all the datasets.")
    group.add_argument("-results_folder", nargs='?', default=results_folder, type=str,
                       help="Folder to put script results (mlflow data, models, etc.).")
    group.add_argument("-logging_folder", nargs='?', default=logging_folder, type=str, help="Folder to put logs.")
    group.add_argument("-no_gather_stats_during_training", action='store_true',
                       help="Gather training statistics from all GPUs.")
    group.add_argument("-no_tqdm", action='store_true', help="Dont show tqdm for every epoch.")

    # Speedup
    group = parser.add_argument_group("Speedup")
    group.add_argument("-no_amp", action='store_true', help="Dont use automatic mixed precision.")
    group.add_argument("-eval_amp", action='store_true', help="Use automatic mixed precision during evaluation")

    # Data loading
    group = parser.add_argument_group("Data loading")
    group.add_argument("-batch_size", nargs='?', default=default_kwargs['batch_size'], type=int,
                       help="Batch size over all graphics cards (togeter).")
    group.add_argument("-num_workers", nargs='?', default=default_kwargs["num_workers"], type=int,
                       help="Number of dataloader worker threads. Should be >0.")
    group.add_argument("-pin_memory", action="store_true", help="Use pin_memory of torch Dataloader.")
    group.add_argument("-prefetch_factor", nargs='?', default=default_kwargs["prefetch_factor"], type=int,
                       help="Prefetch factor for dataloader workers (how many batches to fetch)")
    group.add_argument("-no_shuffle", action="store_true", help="Don't shuffle the training data.")

    # Optimizer
    group = parser.add_argument_group("Optimizer")
    group.add_argument("-opt", nargs='?', default=default_kwargs['opt'], type=str, help="Optimizer to use.")
    group.add_argument("-weight_decay", nargs='?', default=default_kwargs["weight_decay"], type=float,
                       help="Weight decay factor for use in AdamW/LAMB.")
    group.add_argument("-lr", nargs='?', default=default_kwargs["lr"], type=float, help="Initial learning rate.")
    group.add_argument("-max_grad_norm", default=default_kwargs["max_grad_norm"], nargs='?', type=float,
                       help="Maximum norm for the gradients (used for cutoff).")
    group.add_argument("-warmup_epochs", nargs='?', default=default_kwargs["warmup_epochs"], type=int,
                       help="Number of epochs of linear warmup.")
    group.add_argument("-label_smoothing", nargs='?', default=default_kwargs["label_smoothing"], type=float,
                       help="Label smoothing factor.")
    group.add_argument("-sched", nargs='?', default=default_kwargs["sched"], choices=['cosine', 'const'],
                       help="Learning rate schedule.")
    group.add_argument("-min_lr", nargs='?', default=default_kwargs["min_lr"], type=float,
                       help="Minimum learning rate to be hit by scheduler.")
    group.add_argument("-warmup_lr", nargs='?', default=default_kwargs["warmup_lr"], type=float,
                       help="Warmup learning rate.")
    group.add_argument("-warmup_sched", nargs='?', default=default_kwargs["warmup_sched"], choices=['linear', 'const'],
                       help="Schedule for warmup")
    group.add_argument("-opt_eps", nargs='?', default=default_kwargs["opt_eps"], type=float,
                       help="Epsilon value added in the optimizer to stabilize training.")
    group.add_argument("-momentum", nargs='?', default=default_kwargs["momentum"], type=float,
                       help="Optimizer momentum.")

    # Data augmentation
    group = parser.add_argument_group("Data augmentation")
    group.add_argument("-augment_strategy", nargs='?', default=default_kwargs["augment_strategy"], type=str,
                       help="Data augmentation strategy.")
    group.add_argument("-no_aug_flip", action='store_true', help="Turn off data augmentation: horizontal flip.")
    group.add_argument("-no_aug_crop", action='store_true',
                       help="Turn off data augmentation: cropping. This may break the skript.")
    group.add_argument("-no_aug_resize", action='store_true', help="Turn off data augmentation: resize.")
    group.add_argument("-no_aug_grayscale", action='store_true', help="Turn off data augmentation: grayscale.")
    group.add_argument("-no_aug_solarize", action='store_true', help="Turn off data augmentation: solarize.")
    group.add_argument("-no_aug_gauss_blur", action='store_true', help="Turn off data augmentation: gaussian blur.")
    group.add_argument("-no_aug_cutmix", action='store_true', help="Turn off data augmentation: cutmix.")
    group.add_argument("-aug_color_jitter_factor", nargs='?', default=default_kwargs['aug_color_jitter_factor'],
                       type=float, help="Factor to use for the data augmentation: color jitter.")
    group.add_argument("-no_aug_normalize", action='store_true', help="Turn off data augmentation: Normalization")
    group.add_argument("-imsize", nargs='?', default=default_kwargs["imsize"], type=int,
                       help="Image size given to the model -> imsize x imsize.")

    return parser


def partition_choices():
    """Automatically create a list of all possible slurm partitions

    Returns
    -------
    list[str]
        list of partitions
    """

    potential = [l.split(' ')[0] for l in os.popen("sinfo")]
    if len(potential) <= 2:
        return [slurm_defaults['partition']]
    return [p[:-1] if "*" in p else p for p in potential if p != "PARTITION"] + ["A100-SDS,A100-40GB"]


def slurm_parser(parser=None):
    """Add srun arguments to the given parser

    Parameters
    ----------
    parser : argparse.ArgumentParser
        base parser to extend; default is parser from *base_parser*

    Returns
    -------
    argparse.ArgumentParser
        parser
    """

    if parser is None:
        parser = base_parser()
    group = parser.add_argument_group("Slurm arguments")
    group.add_argument("-partition", nargs='?', default=slurm_defaults["partition"], choices=partition_choices(),
                       help="Slurm partition to use")
    group.add_argument("-container-image", nargs='?', default=slurm_defaults["container-image"], type=str,
                       help="Path to slurm container image (.sqsh)")
    group.add_argument("-container-workdir", nargs='?', default=slurm_defaults["container-workdir"], type=str,
                       help="Working directory in container")
    group.add_argument("-container-mounts", nargs='?', default=slurm_defaults["container-mounts"], type=str,
                       help="All slurm mounts separated by ','.")
    group.add_argument("-job-name", nargs='?', default=slurm_defaults["job-name"], type=str,
                       help="Slurm job name. Will default to '<model> <task>'.")
    group.add_argument("-nodes", nargs='?', default=slurm_defaults["nodes"], type=int,
                       help="Number of cluster nodes to use.")
    group.add_argument("-ntasks", nargs='?', default=slurm_defaults["ntasks"], type=int,
                       help="Number of GPUs to use for the job.")
    group.add_argument("-cpus-per-task", nargs='?', default=slurm_defaults["cpus-per-task"], type=int,
                       help="Number of CPUs per task/GPU.")
    group.add_argument("-mem-per-gpu", nargs='?', default=slurm_defaults["mem-per-gpu"], type=int,
                       help="Ram per GPU (in Gb) to use. Will be given as total mem in srun command.")
    group.add_argument("-task-prolog", nargs='?', default=slurm_defaults["task-prolog"], type=str,
                       help="Shell script for task prolog (installing packages, etc.).")

    group = parser.add_argument_group("Run locally")
    group.add_argument("-local", action='store_true', help="Run locally; not in slurm", default=False)

    return parser


def parse_args(args=None):
    """Parse args from *base_parser* and insert defaults

    Returns
    -------
    DotDict
        parsed args
    """

    if args is None:
        parser = base_parser()
        args = parser.parse_args()
        args = dict(vars(args))

    parsed_args = {}
    for key, val in args.items():
        if key.startswith("no_"):
            parsed_args[key[3:]] = not val
        else:
            parsed_args[key] = val
    return parsed_args


def inside_slurm():
    """Test for being inside a slurm container.

    Works by testing for environment variable 'RANK'.

    Returns
    -------
    bool
        true if inside slurm container, false if outside slurm container
    """

    return "RANK" in os.environ


# TODO: fix ./runscript.tmp: 18: Syntax error: Unterminated quoted string
def create_runscript(args, file_name="runscript.tmp"):
    """
    Create a run script for a distributed training job using SLURM.

    Parameters
    ----------
    args : dict
        A dictionary containing various arguments for the job, including parameters for SLURM and for training.
    file_name : str, optional
        The name of the file to create. Defaults to "runscript.tmp".

    Returns
    -------
    str
        The name of the created file.

    Examples
    --------
    >>> args = {"model": "vit_large_patch16_384", "task": "pre-train", "batch_size": 256, ...}
    >>> file_name = "my_run_script.sh"
    >>> create_runscript(args, file_name)
    """

    task_args = ""
    slurm_command = "echo run distributed:\necho python3 main.py {0}\n\nsrun -K \\\n  --gpus-per-task=1 \\\n  --gpu-bind=none \\\n"
    for key, val in args.items():
        if key == "local":
            continue
        if key.replace("_", "-") in slurm_defaults:
            # it's a parameter for srun
            # slurm has - instead of _
            key = key.replace("_", "-")
            if key == "mem-per-gpu":
                # convert mem-per-gpu to mem
                slurm_command += f"  --mem={val * args['ntasks']}G \\\n"
                continue
            if key == "job-name" and val is None:
                # default jobname is '<task> <model>'
                model_str = args['model']
                task = args['task']
                if task == 'pre-train':
                    # it's just the model name...
                    model = model_str.split('_')[0]
                else:
                    # it's a path to the tar file
                    res_folder = args['results_folder'] + '/models/'
                    if not model_str.startswith(res_folder):
                        model = "<vit model>"
                    else:
                        model = model_str[len(res_folder):].split('_')[1].split(' ')[0]
                val = f"\"{task} {model}\""
            if key == "task-prolog" and val is None:
                continue
            slurm_command += f"  --{key}={val} \\\n"
        else:
            # it's a parameter for the training
            if val is None:
                continue
            if key in default_kwargs and val == default_kwargs[key]:
                continue
            if key in ["dataset_root", "results_folder", "logging_folder"] and val == globals()[key]:
                continue
            # only note params that are not the default value
            if key.startswith("no_"):
                # 'no_' inverts the value
                if val == default_kwargs[key[3:]]:
                    # Flag was set
                    task_args += f"-{key} "
                continue
            if isinstance(val, bool):
                task_args += f"-{key} "
                continue
            if isinstance(val, str):
                task_args += f"-{key} \"{val}\" "
            else:
                task_args += f"-{key} {val} "

    slurm_command += "python3 main.py {0}\n"
    os.umask(0)  # make it possible to create an executable file
    with open(file_name, "w+", opener=lambda pth, flgs: os.open(pth, flgs, 0o777)) as f:
        f.write(slurm_command.format(task_args))
    return file_name


def main():
    if not inside_slurm():
        # Make execution script and execute it
        parser = slurm_parser()
        args = vars(parser.parse_args())
        if args['task'] in ['pre-train', 'fine-tune']:
            if 'run_name' not in args or args['run_name'] is None or len(args['run_name']) == 0:
                parser.error(f"-run_name is required for task {args['task']}")

        if not args['local']:
            script_name = create_runscript(args)
            os.system('./' + script_name)  # run srun to execute this script in slurm cluster
                                           # -> the following lines will be executed there
            exit(0)

        # local execution is wanted
        for key in list(args.keys()):
            if key.replace("_", "-") in slurm_defaults.keys():
                args.pop(key)
        args = parse_args(args)

    else:
        args = parse_args()

    if args["task"] == "pre-train":
        if 'dataset' not in args or args['dataset'] is None:
            args['dataset'] = 'imagenet21k'
        assert 'epochs' in args and args['epochs'] is not None, f"How many epochs should I train for?"
        from train import pretrain
        pretrain(**args)
    elif args["task"] == "fine-tune":
        if 'dataset' not in args or args['dataset'] is None:
            args['dataset'] = 'imagenet'
        assert 'epochs' in args and args['epochs'] is not None, f"How many epochs should I train for?"
        from train import finetune
        finetune(**args)
    elif args["task"] == "parser-test":
        print(args)
    elif args["task"] == "eval-metrics":
        if 'dataset' not in args or args['dataset'] is None:
            args['dataset'] = 'CIFAR10'
        from evaluate import evaluate_metrics
        evaluate_metrics(**args)
    elif args["task"] == "eval":
        if 'dataset' not in args or args['dataset'] is None:
            args['dataset'] = 'ImageNet'
        from evaluate import evaluate
        evaluate(**args)
    elif args["task"] == "continue":
        from recover import continue_training
        continue_training(**args)
    else:
        raise NotImplementedError(f"Task {args['task']} is not implemented.")


if __name__ == "__main__":
    main()
