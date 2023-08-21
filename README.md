# Which Transformer to Favor: <br>A Comparative Analysis of Efficiency in Vision Transformers
![First plot from the paper: Pareto front of throughput vs. accuracy](./figures/throughput_vs_acc_size_imsize.png)

This is the code for the paper [Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers](https://arxiv.org/abs/2308.09372), a benchmark of over 30 different efficient vision trainsformers.
We train models from scratch and track multiple efficiency metrics. You can use [this website](https://transformer-benchmark.github.io) to interactively explore the data.

### Abstract
The growing popularity of Vision Transformers as the go-to models for image classification has led to an explosion of architectural modifications claiming to be more efficient than the original ViT.
However, a wide diversity of experimental conditions prevents a fair comparison between all of them, based solely on their reported results.
That is why we conduct an independent and comprehensive analysis of more than 30 models to evaluate the efficiency of vision transformers and related architectures, considering various performance metrics and conducting our own measurements.
This way we provide fair, unbiased baselines and metrics, all in one place, therefore enabling practitioners to make more informed decisions.
Our findings highlight, that ViT is still Pareto optimal across multiple efficiency metrics, as well as the efficiency of sequence reducing approaches. 
We find that hybrid attention-CNN models fare especially well, when it comes to low inference memory and number of parameters and also show that it is better to scale the model size, than the image size.
Furthermore, we uncover a strong positive correlation between FLOPS and training memory, which allows for an estimate of VRAM requirements from theoretical measurements only.
		
Thanks to our holistic evaluation, this study offers valuable insights for practitioners and researchers, facilitating informed decisions when selecting models for specific applications.

## Requirements
This project heavily builds on [timm](https://github.com/huggingface/pytorch-image-models) and open source implementations of the models that are tested.
All requirements are listed in [requirements.txt](./requirements.txt).
To install those, run
```commandline
pip install -r requirements.txt
```

## Usage
After **cloning this repository**, you can train and test a lot of different models.
By default, a `srun` command is executed to run the code on a slurm cluster. 
To run on the local machine, append the `-local` flag to the command.

### Dataset Preparation
Supported datasets are CIFAR10, ImageNet-21k, and ImageNet-1k.

The CIFAR10 dataset has to be located in a subfolder of the dataset root directory called `CIFAR`.
This is the normal `CIFAR10` from `torchvision.datasets`.

To speed up the data loading, the ImageNet datasets are read using [`datadings`](https://datadings.readthedocs.io/en/stable/). 
The `.msgpack` files for **ImageNet-1k** should be located in `<dataset_root_folder>/imagenet/msgpack`, 
while the ones for **ImageNet-21k** should be in `<dataset_root_folder>/imagenet-21k`.
See the [datadings documentation](https://datadings.readthedocs.io/en/stable/) for information on how to create those files.

### Training
#### Pretraining
To pretrain a model on a given dataset, run
```commandline
./main.py -model <model_name> -epochs <epochs> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This will save a checkpoint (`.tar` file) every `<save_epochs>` epochs (the default is 10), which contains all the model weights, along with the optimizer and scheduler state, and the current training stats.
The default pretraining dataset is ImageNet-21k.

#### Finetuning
A model (checkpoint) can be finetuned on another dataset using the following command:
```commandline
./main.py -task fine-tune -model <model_checkpoint_file.tar> -epochs <epochs> -lr <lr> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This will also save new checkpoints during training.
The default finetuning dataset is ImageNet-1k.

### Evaluation
It is also possible to evaluate the models.
To evaluate the model's accuracy and the efficiency metrics, run
```commandline
./main.py -task eval -model <model_checkpoint_file.tar> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
The default evaluation dataset is ImageNet-1k.

To only evaluate the efficiency metrics, run
```commandline
./main.py -task eval-metrics -model <model_checkpoint_file.tar> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This utilizes the CIFAR10 dataset by default.

### Further Arguments
There can be multiple further arguments and flags given to the scripts.
The most important ones are

| Arg                                | Description                                            |
|:-----------------------------------|:-------------------------------------------------------|
| `-model <model>`                   | Model name or checkpoint.                              |
| `-run_name <name for the run>`     | Name or description of this training run.              |
| `-dataset <dataset>`               | Specifies a dataset to use.                            |
| `-task <task>`                     | Specifies a task. The default is `pre-train`.          |
| `-local`                           | Run on the local machine, not on a slurm cluster.      |
| `-dataset_root <dataset root>`     | Root folder of the datasets.                           |
| `-results_folder <results folder>` | Folder to save results into.                           |
| `-logging_folder <logging folder>` | Folder for saving logfiles.                            |
| `-epochs <epochs>`                 | Epochs to train.                                       |
| `-lr <lr>`                         | Learning rate. Default is 3e-3.                        |
| `-batch_size <bs>`                 | Batch size. Default is 2048.                           |
| `-weight_decay <wd>`               | Weight decay. Default is 0.02.                         |
| `-imsize <image resolution>`       | Resulution of the image to train with. Default is 224. |

For a list of all arguments, run
```commandline
./main.py --help
```

## Supported Models
These are the models we support. Links are to original code sources. If no link is provided, we implemented the architecture from scratch, following the specific paper.

| Architecture                                                                                                                                        | Versions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|:----------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [AViT](https://github.com/NVlabs/A-ViT/blob/master/timm/models/act_vision_transformer.py)                                                           | `avit_tiny_patch16`, `avit_small_patch16`, `avit_base_patch16`, `avit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                                                           |
| [CaiT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cait.py)                                                           | `cait_xxs24`, `cait_xxs36`, `cait_xs24`, `cait_s24`, `cait_s36`, `cait_m36`, `cait_m48`                                                                                                                                                                                                                                                                                                                                                                                                        |
| [CoaT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/coat.py)                                                           | `coat_tiny`, `coat_mini`, `coat_small`, `coat_lite_tiny`, `coat_lite_mini`, `coat_lite_small`, `coat_lite_medium`                                                                                                                                                                                                                                                                                                                                                                              |
| [CvT](https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py)                                                                             | `cvt_13`, `cvt_21`, `cvt_w24`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [DeiT](https://github.com/facebookresearch/deit)                                                                                                    | `deit_tiny_patch16_LS`, `deit_small_patch16_LS`, `deit_medium_patch16_LS`, `deit_base_patch16_LS`, `deit_large_patch16_LS`, `deit_huge_patch14_LS`, `deit_huge_patch14_52_LS`, `deit_huge_patch14_26x2_LS`, `deit_Giant_48_patch14_LS`, `deit_giant_40_patch14_LS`, `deit_small_patch16_36_LS`, `deit_small_patch16_36`, `deit_small_patch16_18x2_LS`, `deit_small_patch16_18x2`, `deit_base_patch16_18x2_LS`, `deit_base_patch16_18x2`, `deit_base_patch16_36x1_LS`, `deit_base_patch16_36x1` |
| [DynamicViT](https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py)                                                                 | `dynamic_vit_tiny_patch16`, `dynamic_vit_90_tiny_patch16`, `dynamic_vit_70_tiny_patch16`, `dynamic_vit_small_patch16`, `dynamic_vit_90_small_patch16`, `dynamic_vit_70_small_patch16`, `dynamic_vit_base_patch16`, `dynamic_vit_70_base_patch16`, `dynamic_vit_90_base_patch16`, `dynamic_vit_large_patch16`, `dynamic_vit_90_large_patch16`, `dynamic_vit_70_large_patch16`                                                                                                                   |
| [EfficientFormerV2](https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer_v2.py)                                        | `efficientformerv2_s0`, `efficientformerv2_s1`, `efficientformerv2_s2`, `efficientformerv2_l`                                                                                                                                                                                                                                                                                                                                                                                                  |
| [EViT](https://github.com/youweiliang/evit)                                                                                                         | `evit_tiny_patch16`, `evit_tiny_patch16_fuse`, `evit_small_patch16`, `evit_small_patch16_fuse`, `evit_base_patch16`, `evit_base_patch16_fuse`                                                                                                                                                                                                                                                                                                                                                  |
| FNet                                                                                                                                                | `fnet_vit_tiny_patch16`, `fnet_vit_small_patch16`, `fnet_vit_base_patch16`, `fnet_vit_large_patch16`, `fnet_vit_tiny_patch4`, `fnet_vit_small_patch4`, `fnet_vit_base_patch4`, `fnet_vit_large_patch4`                                                                                                                                                                                                                                                                                         |
| [FocalNet](https://github.com/microsoft/FocalNet/blob/main/classification/focalnet.py)                                                              | `focalnet_tiny_srf`, `focalnet_small_srf`, `focalnet_base_srf`, `focalnet_tiny_lrf`, `focalnet_small_lrf`, `focalnet_base_lrf`, `focalnet_tiny_iso`, `focalnet_small_iso`, `focalnet_base_iso`, `focalnet_large_fl3`, `focalnet_large_fl4`, `focalnet_xlarge_fl3`, `focalnet_xlarge_fl4`, `focalnet_huge_fl3`, `focalnet_huge_fl4`                                                                                                                                                             |
| [GFNet](https://github.com/raoyongming/GFNet)                                                                                                       | `gfnet_tiny_patch4`, `gfnet_extra_small_patch4`, `gfnet_small_patch4`, `gfnet_base_patch4`, `gfnet_tiny_patch16`, `gfnet_extra_small_patch16`, `gfnet_small_patch16`, `gfnet_base_patch16`                                                                                                                                                                                                                                                                                                     |
| [HaloNet](https://github.com/lucidrains/halonet-pytorch)                                                                                            | `halonet_h0`, `halonet_h1`, `halonet_h2`                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Linear Transformer                                                                                                                                  | `linear_vit_tiny_patch16`, `linear_vit_small_patch16`, `linear_vit_base_patch16`, `linear_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                                   |
| [Linformer](https://github.com/lucidrains/linformer)                                                                                                | `linformer_vit_tiny_patch16`, `linformer_vit_small_patch16`, `linformer_vit_base_patch16`, `linformer_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                       |
| [MLP-Mixer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mlp_mixer.py)                                                 | `mixer_s32`, `mixer_s16`, `mixer_b32`, `mixer_b16`, `mixer_l32`, `mixer_l16`                                                                                                                                                                                                                                                                                                                                                                                                                   |
| [NyströmFormer](https://github.com/mlpen/Nystromformer/blob/main/ImageNet/T2T-ViT/models/token_nystromformer.py)                                    | `nystrom64_vit_tiny_patch16`, `nystrom32_vit_tiny_patch16`, `nystrom64_vit_small_patch16`, `nystrom32_vit_small_patch16`, `nystrom64_vit_base_patch16`, `nystrom32_vit_base_patch16`, `nystrom64_vit_large_patch16`, `nystrom32_vit_large_patch16`                                                                                                                                                                                                                                             |
| [Permormer](https://github.com/lucidrains/performer-pytorch)                                                                                        | `performer_vit_tiny_patch16`, `performer_vit_small_patch16`, `performer_vit_base_patch16`, `performer_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                       |
| PolySA                                                                                                                                              | `polysa_vit_tiny_patch16`, `polysa_vit_small_patch16`, `polysa_vit_base_patch16`, `polysa_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                                   |
| [ResNet](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py)                                                       | `resnet18`, `resnet34`, `resnet26`, `resnet50`, `resnet101`, `wide_resnet50_2`                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [Routing Transformer](https://github.com/lucidrains/routing-transformer)                                                                            | `routing_vit_tiny_patch16`, `routing_vit_small_patch16`, `routing_vit_base_patch16`, `routing_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                               |
| [Sinkhorn Transformer](https://github.com/lucidrains/sinkhorn-transformer)                                                                          | `sinkhorn_cait_tiny_bmax32_patch16`, `sinkhorn_cait_tiny_bmax64_patch16`, `sinkhorn_cait_small_bmax32_patch16`, `sinkhorn_cait_small_bmax64_patch16`, `sinkhorn_cait_base_bmax32_patch16`, `sinkhorn_cait_base_bmax64_patch16`, `sinkhorn_cait_large_bmax32_patch16`, `sinkhorn_cait_large_bmax64_patch16`                                                                                                                                                                                     |
| [STViT](https://github.com/changsn/STViT-R/blob/main/models/swin_transformer.py)                                                                    | `stvit_swin_tiny_p4_w7`, `stvit_swin_base_p4_w7`                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [Swin](https://github.com/microsoft/Swin-Transformer)                                                                                               | `swin_tiny_patch4_window7`, `swin_small_patch4_window7`, `swin_base_patch4_window7`, `swin_large_patch4_window7`                                                                                                                                                                                                                                                                                                                                                                               |
| [SwinV2](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)                                                     | `swinv2_tiny_patch4_window7`, `swinv2_small_patch4_window7`, `swinv2_base_patch4_window7`, `swinv2_large_patch4_window7`                                                                                                                                                                                                                                                                                                                                                                       |
| [Switch Transformer](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py) | `switch_8_vit_tiny_patch16`, `switch_8_vit_small_patch16`, `switch_8_vit_base_patch16`, `switch_8_vit_large_patch16`                                                                                                                                                                                                                                                                                                                                                                           |
| [Synthesizer](https://github.com/10-zin/Synthesizer/blob/master/synth/synthesizer/modules.py)                                                       | `synthesizer_fd_vit_tiny_patch16`, `synthesizer_fr_vit_tiny_patch16`, `synthesizer_fd_vit_small_patch16`, `synthesizer_fr_vit_small_patch16`, `synthesizer_fd_vit_base_patch16`, `synthesizer_fr_vit_base_patch16`, `synthesizer_fd_vit_large_patch16`, `synthesizer_fr_vit_large_patch16`                                                                                                                                                                                                     |
| [TokenLearner](https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py)                                          | `token_learner_vit_8_50_tiny_patch16`, `token_learner_vit_8_75_tiny_patch16`, `token_learner_vit_8_50_small_patch16`, `token_learner_vit_8_75_small_patch16`, `token_learner_vit_8_50_base_patch16`, `token_learner_vit_8_75_base_patch16`, `token_learner_vit_8_50_large_patch16`, `token_learner_vit_8_75_large_patch16`                                                                                                                                                                     |
| [ToMe](https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py)                                                                       | `tome_vit_tiny_r8_patch16`, `tome_vit_tiny_r13_patch16`, `tome_vit_small_r8_patch16`, `tome_vit_small_r13_patch16`, `tome_vit_base_r8_patch16`, `tome_vit_base_r13_patch16`, `tome_vit_large_r8_patch16`, `tome_vit_large_r13_patch16`                                                                                                                                                                                                                                                         |
| [ViT](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)                                              | `ViT-{Ti,S,B,L}/<patch_size>`                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [Wave ViT](https://github.com/YehLi/ImageNetModel/blob/main/classification/wavevit.py)                                                              | `wavevit_s`, `wavevit_b`, `wavevit_l`                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [XCiT](https://github.com/facebookresearch/xcit)                                                                                                    | `xcit_nano_12_p16`, `xcit_tiny_12_p16`, `xcit_small_12_p16`, `xcit_tiny_24_p16`, `xcit_small_24_p16`, `xcit_medium_24_p16`, `xcit_large_24_p16`, `xcit_nano_12_p8`, `xcit_tiny_12_p8`, `xcit_small_12_p8`, `xcit_tiny_24_p8`, `xcit_small_24_p8`, `xcit_medium_24_p8`, `xcit_large_24_p8`                                                                                                                                                                                                      |


## License
We release this code under the [MIT license](./LICENSE).

## Citation
If you use this codebase in your project, please cite
```BibTeX
@misc{Nauen2023WTFBenchmark,
      title={Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers}, 
      author={Tobias Christian Nauen and Sebastian Palacio and Andreas Dengel},
      year={2023},
      eprint={2308.09372},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
