"""
Module to load the datasets, using torch and datadings.
"""
from random import uniform

import torch
from PIL import ImageFilter
from torchvision.datasets import CIFAR10, CIFAR100
from datadings.reader import MsgpackReader
from datadings.torch import CompressedToPIL, Compose, Dataset
import torchvision.transforms as tv_transforms
from types import MethodType
import msgpack
from collections import abc
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import Module
from torchvision.transforms import Resize, InterpolationMode, CenterCrop, RandomCrop, RandomHorizontalFlip, Grayscale, \
    RandomSolarize, RandomChoice, ColorJitter, Normalize, ToTensor


class IndividualAugmenter(torch.nn.Module):
    """
    Use the augmentation function on each image of a batch individually, instead of all at once.
    This makes the random decisions for each image different, instead of the same.
    """

    def __init__(self, aug_function):
        """
        Parameters
        ----------
        aug_function : abc.Callable
            augmentation function for a singe image (as a channels x width x height tensor)
        """

        super().__init__()
        self.aug_function = aug_function

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            image or batch of images as tensors

        Returns
        -------
        torch.Tensor
            augmented image or batch of augmented images
        """
        if len(x.shape) == 3:
            # it's a single image
            return self.aug_function(x)

        # it's a batch of images
        return torch.stack([self.aug_function(x_i) for x_i in x], dim=0)


def data_augmentation(args, as_list=False, test=False):
    """Create the data augmentation.

    Parameters
    ----------
    args : arguments
    as_list : bool
        return list of transformations, not composed transformation

    Returns
    -------
    torch.nn.Module | list[torch.nn.Module]
        composed transformation of list of transformations
    """

    augs = []
    if args.aug_resize:
        augs.append(Resize(args.imsize, interpolation=InterpolationMode.BICUBIC))

    if test and args.aug_crop:
        augs.append(CenterCrop(args.imsize))
    elif args.aug_crop:
        augs.append(RandomCrop(args.imsize, padding=4, padding_mode="reflect"))

    if not test:
        if args.aug_flip:
            augs.append(RandomHorizontalFlip(p=.5))

        augs_choice = []
        if args.aug_grayscale:
            augs_choice.append(Grayscale(num_output_channels=3))
        if args.aug_solarize:
            augs_choice.append(RandomSolarize(threshold=128, p=1.))
        if args.aug_gauss_blur:
            # TODO: check kernel size?
            # augs_choice.append(GaussianBlur(kernel_size=7))
            augs_choice.append(QuickGaussBlur())

        if len(augs_choice) > 0:
            augs.append(RandomChoice(augs_choice))

        if args.aug_color_jitter_factor > 0.:
            augs.append(
                ColorJitter(args.aug_color_jitter_factor, args.aug_color_jitter_factor, args.aug_color_jitter_factor))

    augs.append(ToTensor())
    if args.aug_normalize:
        augs.append(Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))

    if as_list:
        return augs
    return IndividualAugmenter(Compose(augs))


class QuickGaussBlur:
    """
    Gaussian blur transformation using PIL ImageFilter
    """
    def __init__(self, sigma=(.2, 2.)):
        """

        Parameters
        ----------
        sigma : tuple[float, float]
            range of sigma for blur
        """
        self.sigma_min, self.sigma_max = sigma

    def __call__(self, img):
        """

        Parameters
        ----------
        img : PIL.Image
            image

        Returns
        -------
        PIL.Image
            blured image
        """
        return img.filter(ImageFilter.GaussianBlur(radius=uniform(self.sigma_min, self.sigma_max)))


class RemoveTransform:
    """
    Remove data from transformation.
    To use with default collate function.
    """

    def __call__(self, x, y=None):
        if y is None:
            return [1]
        return [1], y

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def collate_imnet(data):
    """Custom collate function for imagenet(1k / 21k) with datadings

    Parameters
    ----------
    data : list[dict[str, Any]]
        images for a batch

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        images, labels
    """

    if isinstance(data[0]['image'], torch.Tensor):
        ims = torch.stack([d['image'] for d in data], dim=0)
    else:
        ims = [d['image'] for d in data]
    labels = torch.tensor([d['label'] for d in data])
    # keys = [d['key'] for d in data]
    return ims, labels  # , keys


def prepare_dataset(dataset_name, args, transform=None, train=True, rank=None):
    """Load a dataset from disk, different formats are used for different datasets.

    Supported datasets: CIFAR10, ImageNet, ImageNet21k

    Parameters
    ----------
    dataset_name : str
        name of the dataset
    args
        further arguments
    transform : list[Module] | str
        transformations to use on the data; the list gets composed, or give args.augment_strategy
    train : bool
        use the training split (or test/validation split)
    rank : int
        global rank of this process in distributed training

    Returns
    -------
    tuple[DataLoader, int]
        the data loader, and number of classes
    """
    if transform is None:
        if args.augment_strategy == '3-augment':
            transform = data_augmentation(args, as_list=True, test=not train)
        else:
            raise NotImplementedError(f"Augmentation strategy {args.augment_strategy} is not implemented (yet).")

    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        dataset = CIFAR10(root=args.dataset_root + "CIFAR", train=train, download=False, transform=tv_transforms.Compose(transform))
        n_classes, collate = 10, None

    elif dataset_name in ['imagenet', 'imagenet21k']:
        reader = MsgpackReader(f"{args.dataset_root}imagenet/msgpack/{'train' if train else 'val'}.msgpack")
        if '21k' in dataset_name:
            reader = MsgpackReader(f"{args.dataset_root}imagenet21k/{'train' if train else 'val'}.msgpack")

        dataset = Dataset(reader, transforms={'image': Compose([CompressedToPIL()] + transform)})

        n_classes, collate = 1000, collate_imnet
        if '21k' in dataset_name:
            n_classes = 10_450

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented (yet).")

    if args.distributed:
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=train and args.shuffle)
    else:
        sampler = None

    data_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=args.pin_memory,
                             num_workers=args.num_workers, drop_last=train,
                             prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
                             persistent_workers=(args.num_workers > 0), collate_fn=collate,
                             shuffle=None if sampler else train and args.shuffle, sampler=sampler)
    return data_loader, n_classes



