import math
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional

import torch
from torch import optim
import torchvision.transforms as T

from geossl import BYOL, MoCo, SimCLR, barlowtwins, ResNetBackbone
from geossl.augmentations import SimCLRAugmentations, AugmentationSpecs
from geossl.augmentations import (
    SimCLRAugmentations,
    AugmentationSpecs,
) 
import geossl.parser as geoparser
from geossl.utils import is_part_in_use
from geossl.datasets import get_dataset_spec, create_dataset
from geossl.optimizer import LARS

@geoparser.dataparser
class TrainingArgs:
    train_dir: Path = geoparser.Field(positional=True, help="The Training dataset path")

    backbone_arch: str = geoparser.Field(
        default="resnet18", choices=["resnet18", "resnet50"]
    )
    method: str = geoparser.Field(
        default="simclr", choices=["simclr", "barlow", "byol", "moco"]
    )
    temperature: float = geoparser.Field(
        default=0.07, help="Temperature for the nt_xent loss [default=0.07]"
    )
    optimizer: str = geoparser.Field(default="sgd", choices=["sgd", "lars"])

    n_gpus: int = 1
    n_epochs: int = 10
    batch_size: int = 16
    checkpoint_dir: Path = Path("checkpoint/")
    weight_decay: float = 1e-4
    learning_rate_weights: float = 0.2
    learning_rate_biases: float=0.0048
    cosine: bool = geoparser.Field(action="store_true")
    no_small_conv: bool = geoparser.Field(action="store_true")
    augmentation_specs: Optional[str] = None

def adjust_learning_rate(args, optimizer, loader, step):
    if args.cosine:
        max_steps = args.n_epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = args.batch_size / 256
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else: 
            step -= warmup_steps:
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_group[1]["lr"] = lr * args.learning_rate_biases

    else: 
        lr_decay_steps = [700, 800, 900]
        lr_decay_rate = 0.1

        n_epochs = step // len(loader)
        steps = sum(n_epochs > x for x in lr_decay_steps)
        
        lr = lr_decay_rate ** steps
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases

def main_worker(device: torch.device, args: TrainingArgs):
    is_distributed = hasattr(args, "rank")
    torch.manual_seed(42)

    if is_distributed:
        assert device != "cpu", "Cannot use distributed with cpu"
        args.rank += device
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            rank=args.rank,
            world_size=args.n_gpus,
        )
    if device != "cpu":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    if "resisc" in str(args.train_dir):
        dataset_id = "resisc"
    elif "eurosat_rgb" in str(args.train_dir):
        dataset_id = "eurosat_rgb"
    elif "eurosat" in str(args.train_dir):
        dataset_id = "eurosat"
    else: 
        raise NotImplementedError()
    

    dataset_spec = get_dataset_spec(dataset_id)
    img_size, crop_size = dataset_spec, dataset_spec.crop_size

    if args.augmentation_specs is not None:
        aug_specs = AugmentationSpecs.from_str(args.augmentation_specs)
    else:
        aug_specs = AugmentationSpecs()
    augment = SimCLRAugmentations(
        size=crop_size,
        mean=dataset_spec.mean,
        std=dataset_spec.mean,
        std=dataset_spec.std,
        specs=aug_specs,
        move_to_tensor=dataset_id == "eurosat_rgb",
    )

    img_transform = T.Compose([T.Resize(img_size), T.CenterCrop(crop_size), augment,])

    print("creating dataset" + str(args.train_dir))
    train_dataset = create_dataset(args.train_dir, train=True, transform=img_transform,)
    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if is_distributed
        else None
    )

    num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() // 4))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpus,
        sampler=sampler,
        shuffle=not is_distributed,
        num_workers=num_workers,
        pin_memory=device !="cpu",
        persistent_workers=False,
    )

    small_conv = img_size < 100 and not args.no_small_conv
    backbone = ResNetBackbone(args.backbone_arch, small_conv=small_conv)
    if args.method == "simclr":
        model = SimCLR(backbone, tau=args.temperature)
    elif args.method == "barlow":
        model = barlowtwins(
            backbone, 
            lambd=0.0051,
            batch_size=args.batch_size,
            h_dim=backbone.out_dim * (2 if small_conv else 4),
        )
    elif args.method == "byol":
        encoder_s = backbone
        encoder_t = ResNetBackbone(args.backbone_arch, small_conv=img_size < 100)
        base_momentum, final_momentum = 1.0, 0.99
        model = BYOL(
            encoder_s,
            encoder_t, 
            base_momentum=base_momentum,
            final_momentum=final_momentum,
            n_steps=len(train_loader) * args.n_epochs,
        )
    elif args.method == "moco":
        encoder_s = backbone
        encoder_t = ResNetBackbone(args.backbone_arch, small_conv=img_size < 100)
        base_momentum, final_momentum = 1.0, 0.99
        model = MoCo(
            encoder_s,
            # 176
        )