import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from Myutils.visualize_feature import show_feature_map


class ResidualBlock(nn.Module):
    """ 残差块 (Residual Block) """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        out = self.block(x)
        x = out + x  # 残差连接
        return x

class ResidualSelfAttention(nn.Module):
    """ 先通过 Residual Block，然后通过 Multi-Head Self-Attention """

    def __init__(self, channels, num_heads=8):
        super(ResidualSelfAttention, self).__init__()
        self.res_block = ResidualBlock(channels)
        # self.mhsa = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        # self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # 先通过 Residual Block
        x = self.res_block(x)  # (B, C, H, W)
        # 注意力 -----------------
        # B, C, H, W = x.shape
        # x = x.view(B, C, -1).permute(0, 2, 1)
        # attn_output, _ = self.mhsa(x, x, x)
        # x = self.norm(attn_output)
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        return x
class GetMultiScaleFeature(nn.Module):
    def __init__(self, in_channels):
        super(GetMultiScaleFeature, self).__init__()
        self.init_conv = nn.Conv2d(in_channels, 320, 3, 1, 1)
        self.block1_1 = ResidualSelfAttention(320, 8)
        self.block1_2 = ResidualSelfAttention(320, 8)
        self.up1 = nn.Conv2d(320, 640, kernel_size=1)
        self.block2_1 = ResidualSelfAttention(640, 8)
        self.up2 = nn.Conv2d(640, 1280, kernel_size=1)
        self.block3_1 = ResidualSelfAttention(1280, 8)
        self.block4_1 = ResidualSelfAttention(1280, 8)
        self.block4_2 = ResidualSelfAttention(1280, 8)

    def forward(self, x):
        feature_list = []
        x = self.init_conv(x)  # 320,60,80
        feature_list.append(x)
        x = self.block1_1(x)  # 320,60,80
        feature_list.append(x)
        x = self.block1_2(x)  # 320,60,80
        feature_list.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)  # 320,30,40
        feature_list.append(x)
        x = self.up1(x)  # 640,30,40
        feature_list.append(x)
        x = self.block2_1(x)  # 640,30,40
        feature_list.append(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)  # 640,15,20
        feature_list.append(x)
        x = self.up2(x)  # 1280,15,20
        feature_list.append(x)
        x = self.block3_1(x)  # 1280,15,20
        feature_list.append(x)
        b, c, h, w = x.shape
        if w == 16:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=(8, 10), mode='bilinear', align_corners=False)  # 1280,8,10
        feature_list.append(x)
        x = self.block4_1(x)  # 1280,8,10
        feature_list.append(x)
        x = self.block4_2(x)  # 1280,8,10
        feature_list.append(x)
        return feature_list

class MultiScaleFeatureReconstruct(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureReconstruct, self).__init__()
        self.block10 =nn.Sequential(nn.Conv2d(2560, 1280, 3,1,1),nn.BatchNorm2d(1280),nn.ReLU(inplace=False))
        self.block9 =nn.Sequential(nn.Conv2d(2560, 1280, 3,1,1),nn.BatchNorm2d(1280),nn.ReLU(inplace=False))
        self.block8 =nn.Sequential(nn.Conv2d(2560, 1280, 3,1,1),nn.BatchNorm2d(1280),nn.ReLU(inplace=False))
        self.block7 =nn.Sequential(nn.Conv2d(2560, 640, 3,1,1),nn.BatchNorm2d(640),nn.ReLU(inplace=False))
        self.block6 =nn.Sequential(nn.Conv2d(1280, 640, 3,1,1),nn.BatchNorm2d(640),nn.ReLU(inplace=False))
        self.block5=nn.Sequential(nn.Conv2d(1280, 640, 3,1,1),nn.BatchNorm2d(640),nn.ReLU(inplace=False))
        self.block4=nn.Sequential(nn.Conv2d(1280, 320, 3,1,1),nn.BatchNorm2d(320),nn.ReLU(inplace=False))
        self.block3=nn.Sequential(nn.Conv2d(640, 320, 3,1,1),nn.BatchNorm2d(320),nn.ReLU(inplace=False))
        self.block2=nn.Sequential(nn.Conv2d(640, 320, 3,1,1),nn.BatchNorm2d(320),nn.ReLU(inplace=False))
        self.block1=nn.Sequential(nn.Conv2d(640, 320, 3,1,1),nn.BatchNorm2d(320),nn.ReLU(inplace=False))
        self.block0=nn.Sequential(nn.Conv2d(640, 320, 3,1,1),nn.BatchNorm2d(320),nn.ReLU(inplace=False))
        self.final_conv=nn.Conv2d(320, 4, 3,1,1)
    def forward(self, feature_list):
        x=self.block10(torch.concat((feature_list[10],feature_list[11]), dim=1)) #1280,8,8
        x=self.block9(torch.concat((feature_list[9],x), dim=1))#1280,8,8
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)#1280,16,16
        x=self.block8(torch.concat((feature_list[8],x), dim=1)) #1280,16,16
        x=self.block7(torch.concat((feature_list[7],x), dim=1)) #640,16,16
        x=self.block6(torch.concat((feature_list[6],x), dim=1)) #640,16,16
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)#640,32,32
        x=self.block5(torch.concat((feature_list[5],x), dim=1)) #640,32,32
        x=self.block4(torch.concat((feature_list[4],x), dim=1)) #320,32,32
        x=self.block3(torch.concat((feature_list[3],x), dim=1)) #320,32,32
        x=F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)#320,64,64
        x=self.block2(torch.concat((feature_list[2],x), dim=1))
        x=self.block1(torch.concat((feature_list[1],x), dim=1))
        x=self.block0(torch.concat((feature_list[0],x), dim=1))
        output=self.final_conv(x)
        return output
def compare_model_params(model1, model2, atol=1e-6):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"参数名称不匹配: {name1} vs {name2}")
            return False
        if not torch.allclose(param1, param2, atol=atol):
            print(f"参数 {name1} 不匹配")
            return False
    print("两个模型的参数完全一致")
    return True


if __name__ == '__main__':
    data_list=[
        torch.randn(2,320,64,64),
        torch.randn(2,320,64,64),
        torch.randn(2,320,64,64),
        torch.randn(2,320,32,32),
        torch.randn(2,640,32,32),
        torch.randn(2,640,32,32),
        torch.randn(2,640,16,16),
        torch.randn(2,1280,16,16),
        torch.randn(2,1280,16,16),
        torch.randn(2,1280,8,8),
        torch.randn(2,1280,8,8),
        torch.randn(2,1280,8,8),
    ]
    for i in range(len(data_list)):
        data_list[i]=data_list[i].cuda()
    model=MultiScaleFeatureReconstruct().cuda()
    output=model(data_list)
