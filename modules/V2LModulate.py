import torch
import torch.nn as nn
from Myutils import visualization_tensor
from typing import Optional, List
import torch.nn.functional as F

class V2LModulate_0407a(nn.Module):
    def __init__(self, num_proj_layers: int = 2, clip_v_dim: int = 1024,clip_text_dim:int=768):
        super().__init__()

        layer_list = []
        layer_list += [nn.Linear(clip_v_dim*2, clip_v_dim), nn.LayerNorm(clip_v_dim), nn.LeakyReLU()]
        for i in range(num_proj_layers):
            layer_list +=[nn.Linear(clip_v_dim, clip_v_dim), nn.LayerNorm(clip_v_dim), nn.LeakyReLU()]
        layer_list += [nn.Linear(clip_v_dim, clip_text_dim)]
        self.visual_projection = nn.Sequential(*layer_list)


    def forward(
        self,
        vis_input: Optional[torch.FloatTensor] = None,
        inf_input: Optional[torch.FloatTensor] = None,
        topk:int=3,

    ):
        vis_input_tensor = vis_input.hidden_states[-1][:, 1:, :]
        inf_input_tensor = inf_input.hidden_states[-1][:, 1:, :]

        vis_norm = F.normalize(vis_input_tensor, dim=-1)
        inf_norm = F.normalize(inf_input_tensor, dim=-1)

        cos_sim = (vis_norm * inf_norm).sum(dim=-1)  # (B, 256)

        # 取每个 batch 中 top-3 的索引
        topk_values, topk_indices = cos_sim.topk(k=topk, dim=1)  # topk_indices: (B, 3)

        B, N, C = vis_norm.shape
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, 3, 1024)
        vis_topk_features = torch.gather(vis_input_tensor, dim=1, index=topk_indices_expanded)  # (B, 3, 1024)
        inf_topk_features = torch.gather(inf_input_tensor, dim=1, index=topk_indices_expanded)  # (B, 3, 1024)

        vis_topk_features =torch.concat([vis_topk_features, vis_input.hidden_states[-1][:, 0:1, :]],dim=1)
        inf_topk_features =torch.concat([inf_topk_features, inf_input.hidden_states[-1][:, 0:1, :]],dim=1)

        input=torch.concat([vis_topk_features, inf_topk_features], dim=-1)

        for layer in self.visual_projection:
            input = layer(input)

        return input

if __name__ == '__main__':
    modle=V2LModulate_0407a()
    data1=torch.randn(1, 77, 1024)
    data2=torch.randn(1, 77, 1024)
    output=modle(data1,data2)