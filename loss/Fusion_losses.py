import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(
        0)  # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # window shape: [1,1, 11, 11]
    return window


# 计算 ssim 损失函数
def mssim(img1, img2, window_size=11):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).

    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    # 滤波器窗口
    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret


def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)

    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res


# 方差计算
def std(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1


# def sum(img,  window_size=9):

#     padd = window_size // 2
#     (_, channel, height, width) = img.size()
#     window = create_window(window_size, channel=channel).to(img.device)
#     win1 = torch.ones_like(window)
#     res = F.conv2d(img, win1, padding=padd, groups=channel)
#     return res


def final_ssim(img_ir, img_vis, img_fuse, mask=None):
    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    # std_ir = std(img_ir)
    # std_vi = std(img_vis)
    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    # m = torch.mean(img_ir)
    # w_ir = torch.where(img_ir > m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map1 = map1
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map2 = map2

    ssim = map1 * ssim_ir + map2 * ssim_vi
    # ssim = ssim * w_ir
    return ssim.mean()


def final_mse(img_ir, img_vis, img_fuse):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    w_vi = torch.where(img_ir <= m, one, zero)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    res = map1 * mse_ir + map2 * mse_vi
    res = res * w_vi
    return res.mean()


def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map_ir = torch.where(map1 + mask > 0, one, zero)
    map_vi = 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    return res.mean()


def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1. / (corr + eps)
    return corr_loss


class Sobelxy(nn.Module):
    """
    输入:[b,1,h,w],图片
    输出:2*[b,1,h,w],各个像素的梯度值(x,y方向)(输出保证值为正)
    """
    def __init__(self, device,channel=1):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        # 这里不行就采用expend_dims
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        if channel == 3:
            kernelx = kernelx.repeat(1, 3, 1, 1)
            kernely = kernely.repeat(1, 3, 1, 1)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)

def Fusion_loss(ir, vi, fu, ssim_w=1,grad_w=5,intensity_w=10, device=None):
    # grad_ir =  KF.spatial_gradient(ir, order=2).abs().sum(dim=[1,2])
    # grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1,2])
    # grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1,2])
    # grad_joint = torch.max(grad_ir, grad_vi)
    sobelconv = Sobelxy(device,channel=fu.shape[1])
    vi_grad_x, vi_grad_y = sobelconv(vi)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)

    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)
    ## 梯度损失
    # loss_grad = F.l1_loss(grad_fus, grad_joint)
    ## SSIM损失
    loss_ssim = corr_loss(ir, vi, fu)
    ## 强度损失
    # loss_intensity = final_mse1(ir, vi, fu, mask) + 0 * F.l1_loss(fu, torch.max(ir, vi))
    loss_intensity = F.l1_loss(fu, torch.max(ir, vi))
    loss_total = ssim_w * loss_ssim + grad_w * loss_grad + intensity_w * loss_intensity
    return loss_total, loss_intensity, loss_grad, loss_ssim




class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    计算a,c和b,c之间的关联性,c是融合图片
    """

    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1 * img2) / torch.sqrt(torch.sum(img1 * img1) * torch.sum(img2 * img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)


if __name__ == '__main__':
    data1=torch.rand(1,3,160,160)
    data2 = torch.rand(1, 3, 160, 160)
    data3 = torch.rand(1, 3, 160, 160)
    output=corr_loss(data1,data2,data3)
    print(output)
