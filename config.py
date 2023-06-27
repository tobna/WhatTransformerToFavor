"""
This module contains default configuration parameters.

Attributes:
-----------
default_kwargs (dict): Default hyperparameters for the training process.
slurm_defaults (dict): Default values for SLURM batch job settings.
dataset_root (str): Path to the root directory of the dataset.
results_folder (str): Path to the directory where results are saved.
logging_folder (str): Path to the directory where logs are saved.

"""

default_kwargs = {'seed': None, 'batch_size': 2048, 'aug_flip': True, 'sched': 'cosine', 'min_lr': 1e-5,
                  'warmup_lr': 1e-6, 'aug_crop': True, 'aug_resize': True, 'imsize': 224, 'aug_grayscale': True,
                  'num_workers': 44, 'experiment_name': "EfficientCVBench", 'aug_solarize': True, 'aug_gauss_blur': True,
                  'aug_color_jitter_factor': 0.3, 'weight_decay': 0.02, 'lr': 3e-3, 'max_grad_norm': 1.,
                  'warmup_epochs': 5, 'label_smoothing': 0.1, 'aug_cutmix': True, 'amp': True, 'save_epochs': 10,
                  'pin_memory': False, 'gather_stats_during_training': True, 'prefetch_factor': 2, 'eval_amp': True,
                  'drop_path_rate': 0.05, 'run_name': None, 'aug_normalize': True, 'opt': 'fusedlamb',
                  'layer_scale': True, 'layer_scale_init_values': 1e-4, 'qkv_bias': True, 'pre_norm': False,
                  'warmup_sched': 'linear', 'opt_eps': 1e-7, 'tqdm': True, 'shuffle': True, 'dropout': 0.,
                  'momentum': 0., 'augment_strategy': '3-augment', 'compile_model': True}
                    #, 'model_ema': True, 'model_ema_decay': 0.99996}

slurm_defaults = {'partition': "A100-SDS,A100-80GB", 'container-image': "/netscratch/nauen/images/custom_ViT_v10.sqsh",
                  'container-mounts': '/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`"',
                  "job-name": None, "nodes": 1, "ntasks": 4, "cpus-per-task": 24, "gpus-per-task": 1,
                  "mem-per-gpu": 115, "task-prolog": None, "container-workdir": '"`pwd`"', "time": "5-0"}

dataset_root = "/ds/images/"
results_folder = "/netscratch/nauen/EfficientCVBench"
logging_folder = results_folder + "/logging"
