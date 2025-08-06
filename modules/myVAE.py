import torch.nn as nn
from diffusers.models.vae import Encoder,Decoder
import torch
from Myutils import visualization_tensor
import torch.nn.functional as F
class CustomEncoder(Encoder):
    def __init__(self, config,original_encoder,trainable=False):
        # 继承 Encoder 的初始化
        super().__init__(
            in_channels=config["in_channels"],
            out_channels=config["latent_channels"],
            down_block_types=config["down_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            act_fn=config["act_fn"],
        )
        self.load_state_dict(original_encoder.state_dict(), strict=False)
        if trainable:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x,return_feature=False):
        # original_encoder 的 forward
        sample = self.conv_in(x)
        feature_list=[]
        feature_list.append(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
            feature_list.append(sample)
        sample = self.mid_block(sample)


        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        # ----------------------
        if return_feature:
            return sample,feature_list
        return sample

class CustomDecoder(Decoder):
    def __init__(self, config,original_decoder):
        # 继承 Encoder 的初始化
        super().__init__(
            in_channels=config["latent_channels"],
            out_channels=config["in_channels"],
            up_block_types=config["up_block_types"],
            block_out_channels=config["block_out_channels"],
            layers_per_block=config["layers_per_block"],
            norm_num_groups=config["norm_num_groups"],
            act_fn=config["act_fn"],
        )
        self.load_state_dict(original_decoder.state_dict(), strict=False)
        for param in self.parameters():
            param.requires_grad = False
        self.refine_module_list=nn.ModuleList(
            [
                RefineModule(512,512,1),
                RefineModule(512,512,2),
                RefineModule(256,512,2),
                RefineModule(128,256,2),
            ]
        )
        self.final_refine=RefineModule(128, 128, 1)
    def forward(self, z,x_list=None,latent_embeds=None):
        if x_list is not None:
            x_list=x_list[::-1]
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        if x_list is not None:
            for up_block,refine_module,x in zip(self.up_blocks,self.refine_module_list,x_list):
                sample = refine_module(sample, x)
                sample = up_block(sample, latent_embeds)
        else:
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)


        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)

        if x_list is not None:
            sample = self.final_refine(sample,x_list[-1])
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class ResBlock(nn.Module):
    def __init__(self, in_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        out = identity+ out
        return out

class RefineModule(nn.Module):
    def __init__(self, in_channels, out_channels,scale_factor=2,mode='bilinear',align_corners=False):
        super(RefineModule, self).__init__()
        self.adapter=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        # self.adapter = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_channels),
        # )
        self.scale_factor=scale_factor
        self.mode=mode
        self.align_corners=align_corners
        self.fuse=nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            ResBlock(out_channels,out_channels),
        )



    def forward(self, z,x):
        # visualization_tensor(x, "bchw", num_channels=1)
        x = self.adapter(x)
        x=F.interpolate(x,scale_factor=self.scale_factor,mode=self.mode,align_corners=self.align_corners)
        feat=z+self.fuse(torch.concat((z,x),dim=1))
        return feat