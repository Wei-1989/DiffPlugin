import torch
import torch.nn as nn
from Myutils import visualization_tensor
from typing import Optional, List

class TextAdapter(nn.Module):
    def __init__(self, num_cross_proj_layers: int = 2, clip_v_dim: int = 768):
        super().__init__()

        layer_list = []
        for i in range(num_cross_proj_layers):
            layer_list +=[nn.Linear(clip_v_dim, clip_v_dim), nn.LayerNorm(clip_v_dim), nn.LeakyReLU()]
        layer_list += [nn.Linear(clip_v_dim, 768)]
        self.visual_projection = nn.Sequential(*layer_list)


    def forward(
        self,
        input: Optional[torch.FloatTensor] = None,
    ):
        b,l,c=input.shape
        for layer in self.visual_projection:
            input = layer(input)
            # if isinstance(layer, nn.Linear):
            #     visualization_tensor(projection_input.permute(0, 2, 1).view(b, -1, 16, 16))
        return input