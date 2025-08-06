import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from Myutils.visualize_feature import show_feature_map


class BBasicConv2d(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


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


class BasicFusionBlock(nn.Module):
    def __init__(self, single_in_channels):
        super(BasicFusionBlock, self).__init__()
        self.conv1 = ResidualBlock(single_in_channels * 2)
        self.conv2 = nn.Conv2d(single_in_channels * 2, single_in_channels, 3, 1, 1)

    def forward(self, img1, img2):
        data = torch.cat([img1, img2], dim=1)
        x = self.conv1(data)
        x = self.conv2(x)
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


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=1, k=1):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Second_reduce = nn.Conv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class PSFM(nn.Module):
    def __init__(self, single_in_C: int = 4, out_C: int = 4):
        super(PSFM, self).__init__()
        self.RGBobj = BBasicConv2d(single_in_C, out_C, 3)
        self.Infobj = BBasicConv2d(single_in_C, out_C, 3)
        self.obj_fuse = GEFM(out_C * 2, out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out


class NativeFusion(nn.Module):
    def __init__(self, single_in_C, out_C):
        super(NativeFusion, self).__init__()
        self.img1Proj = BBasicConv2d(single_in_C, out_C, 3)
        self.img2Proj = BBasicConv2d(single_in_C, out_C, 3)
        self.obj_fuse = nn.Conv2d(out_C * 2, out_C, 3, 1, 1)

    def forward(self, img1, img2):
        img1_sum = self.img1Proj(img1)
        img2_sum = self.img2Proj(img2)
        fuse = torch.cat([img1_sum, img2_sum], dim=1)
        out = self.obj_fuse(fuse)
        return out


class CNN_Attention(nn.Module):
    def __init__(self, single_in_C: int = 4, out_C: int = 4):
        super(CNN_Attention, self).__init__()
        self.CNNFusion = NativeFusion(single_in_C, out_C)
        self.AttentionFusion = PSFM(single_in_C, out_C)
        self.fusion = nn.Sequential(
            BBasicConv2d(out_C * 2, out_C * 2),
            nn.Conv2d(out_C * 2, out_C, 3, 1, 1),
        )

    def forward(self, img1, img2):
        cnn_feature = self.CNNFusion(img1, img2)
        attention_feature = self.AttentionFusion(img1, img2)
        out = self.fusion(torch.cat([cnn_feature, attention_feature], dim=1))

        return out


class Channel_Attention(nn.Module):
    def __init__(self, single_in_C: int = 4, out_C: int = 4):
        super(Channel_Attention, self).__init__()
        self.mask_conv = nn.Sequential(
            nn.Conv2d(single_in_C * 2, single_in_C, 3, 1, 1),
            nn.BatchNorm2d(single_in_C),
            nn.LeakyReLU(),
            nn.Conv2d(single_in_C, 1, 3, 1, 1),
        )

    def forward(self, img1, img2):
        mask = torch.sigmoid(self.mask_conv(torch.cat([img1, img2], dim=1)))
        out = mask * img1 + (1 - mask) * img2
        return out


class dual_branch_concat(nn.Module):
    def __init__(self):
        super(dual_branch_concat, self).__init__()

        self.img1_conv1x1 = nn.Conv2d(in_channels=4, out_channels=160, kernel_size=1)

        self.img2_conv1x1 = nn.Conv2d(in_channels=4, out_channels=160, kernel_size=1)

    def forward(self, img1, img2):
        img1 = self.img1_conv1x1(img1)  # (b, 160, 64, 85)

        img2 = self.img2_conv1x1(img2)  # (b, 160, 64, 85)
        fuse = torch.concat((img1, img2), dim=1)
        output_list = [fuse] * 3
        output_list += [0] * 9
        return output_list


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


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super(MultiScaleFusion, self).__init__()
        self.encoder1 = GetMultiScaleFeature(4)
        self.encoder2 = GetMultiScaleFeature(4)
        self.fusion_list = nn.ModuleList([
            BasicFusionBlock(320),
            BasicFusionBlock(320),
            BasicFusionBlock(320),
            BasicFusionBlock(320),
            BasicFusionBlock(640),
            BasicFusionBlock(640),
            BasicFusionBlock(640),
            BasicFusionBlock(1280),
            BasicFusionBlock(1280),
            BasicFusionBlock(1280),
            BasicFusionBlock(1280),
            BasicFusionBlock(1280)
        ])

    def forward(self, img1, img2):
        feature_list = []
        img1_list = self.encoder1(img1)
        img2_list = self.encoder2(img2)
        for img1, img2, fusionBlock in zip(img1_list, img2_list, self.fusion_list):
            fusionBlock.to(img1.device)
            fuse = fusionBlock(img1, img2)
            feature_list.append(fuse)
        return feature_list

class MultiScaleFeatureFusionModule(nn.Module):
    def __init__(self, in_channels_list=[128, 128, 256, 512, 512]):
        super(MultiScaleFeatureFusionModule, self).__init__()
        self.fusion_list = nn.ModuleList()
        for in_channels in in_channels_list:
            self.fusion_list.append(BasicFusionBlock(in_channels))

    def forward(self, img1_list, img2_list):
        feature_list = []
        for img1, img2, fusionBlock in zip(img1_list, img2_list, self.fusion_list):
            # fusionBlock.to(img1.device)
            fuse = fusionBlock(img1, img2)
            feature_list.append(fuse)
        return feature_list
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
    img1 = "../TestData/img1/01458D.png"  # 替换为你的图片路径
    img2 = "../TestData/img2/01458D.png"  # 替换为你的图片路径
    image1 = Image.open(img1).convert("RGB")  # 确保转换为 RGB 格式
    image2 = Image.open(img2).convert("RGB")  # 确保转换为 RGB 格式
    vae = AutoencoderKL.from_pretrained('../pretrained-large-modal/VAE', subfolder="vae", revision=None).to('cuda:0')
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    img1 = vae_image_processor.preprocess(image1, height=480, width=640).to(device=vae.device)  # image now is tensor in [-1,1]
    img2 = vae_image_processor.preprocess(image2, height=480, width=640).to(device=vae.device)  # image now is tensor in [-1,1]
    img1_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img1)), 2, dim=1)[0]
    img2_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img2)), 2, dim=1)[0]

    model3 = MultiScaleFusion().to('cuda:0')
    model4 = MultiScaleFusion().to('cuda:0')
    model3.load_state_dict(torch.load(r"../results/VIFusion/temp/checkpoint-4/vae_fuse_net.pt", weights_only=True)['model'],
                           strict=True)
    model4.load_state_dict(torch.load(r"../results/VIFusion/temp/checkpoint-4/vae_fuse_net.pt", weights_only=True)['model'],
                           strict=True)
    compare_model_params(model3, model4, atol=1e-6)
    out3 = model3(img1_scb_cond, img2_scb_cond)
    # for i in out3:
    #     show_feature_map(i.detach())
