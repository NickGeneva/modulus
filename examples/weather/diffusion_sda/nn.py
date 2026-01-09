from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from physicsnemo import Module
from physicsnemo.core import ModelMetaData
from physicsnemo.models.diffusion_unets import SongUNetPosEmbd

from physicsnemo.diffusion.multi_diffusion import BasePatching2D

class HRRRSurfaceDiffusionNet(Module):
    """
    Adapter to make a model callable with the correct
    signature ``forward(x, t, condition, **model_kwargs) -> torch.Tensor``.
    """
    # HRRR grid, 16 variables with 4 conditions
    IMG_RESOLUTION = [1059, 1799]
    IMG_CHANNELS = 16 + 4

    def __init__(self, patching: BasePatching2D, use_apex: bool = False):
        super().__init__(meta=ModelMetaData())
        
        # Multi-diffusion paramters, defines how large a single diffusion patch is
        patch_shape = (448, 448)
        patch_num = 4

        self.patching = patching

        # Create model
        channel_mult = [1, 2, 2, 2, 2]
        num_grid_channels = 8
        self.model = SongUNetPosEmbd(
            img_resolution=self.IMG_RESOLUTION,
            in_channels=self.IMG_CHANNELS + num_grid_channels,
            out_channels=self.IMG_CHANNELS,
            N_grid_channels=num_grid_channels,
            gridtype="learnable",
            model_channels=128,
            channel_mult=channel_mult,
            attn_resolutions=[self.IMG_RESOLUTION[0] >> len(channel_mult)],
            use_apex_gn=use_apex,
        )

    
    def forward(self, x: Tensor, sigma: Tensor, condition: Dict[str, Tensor], **model_kwargs):

        x = self.patching.apply(x)
        _, c = next(iter(condition.items()))
        c = self.patching.apply(c)
        
        x = torch.cat([x, c], dim=1)
        global_index = self.patching.global_index(x.shape[0], x.device)

        return self.model(x, sigma, None, global_index=global_index, **model_kwargs)