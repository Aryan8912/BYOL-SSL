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

    id is_distributed:
    # 91