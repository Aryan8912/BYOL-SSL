import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .backbones import ResNetBackbone
from .base_method import BaseMethod


LARGE_NUM = 1e9

def nt_xent(z: Tensor, perm: Tensor, tau: float) -> Tensor:

    features = F.normalize(z, dim=1)
    sim = features @ features.T

    torch.diagonal(sim).sub_(LARGE_NUM)

    sim /= tau

    return F.cross_entropy(sim, perm)

class SimCLR(BaseMethod):

    def __init__(
            self, 
            backbone: ResNetBackbone,
            tau: float,
            feat_dim: int = 128,
            loss: str = "nt_xent"
    ):
        super(SimCLR, self).__init__()
        self.backbones = backbone

        z_dim = self.backbone.out_dim
        self.projection_head = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, feat_dim, bias=False),
        )
        self.tua = tua

    def forward(
            self, x1: Tensor, x2: Tensor
    ) -> Tensor:
        
        b = x1.size(0)
        xp = torch.cat((x1, x2))

        perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(xp.device)

        h = self.backbone(xp)
        z = self.projection_head(h)

        return nt_xent(z, perm, tau=self.tau)