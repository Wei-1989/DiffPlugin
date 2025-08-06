from urllib.error import HTTPError
import numpy as np
import matplotlib.pyplot as plt
import math

from torchvision.utils import make_grid
from PIL import Image, ImageOps
import torch
from typing import Literal
from sklearn.decomposition import PCA
from torchvision import transforms

import time
import random
def safe_show(plt,max_retries=3):
    retries=0
    while retries<max_retries:
        try:
            plt.show()
            break
        except HTTPError as e:

            if e.code == 429:  # Too Many Requests
                wait_time = 2 ** retries + random.uniform(0, 1)  # 指数退避
                print(f"收到HTTP 429错误，等待{wait_time:.2f}秒后重试 ({retries + 1}/{max_retries})")
                time.sleep(wait_time)
                retries += 1
            else:
                raise  # 其他类型的错误直接抛出
def show_feature_map(feature_map, num_channels=1, cmap='turbo', save_path=None,show_info=False,use_pca=True):
    """
    可视化特征图
    :param feature_map: 特征图，[C, H, W]
    :param num_channels: 可视化通道数
    """

    feature_map = feature_map.cpu()
    c, h, w = feature_map.shape
    # 使用 PCA 将 128 维降到 n 维
    if use_pca:
        features_reshaped = feature_map.view(c, -1).T.numpy()  # [480*640, 128]
        pca = PCA(n_components=num_channels)
        features_pca = pca.fit_transform(features_reshaped)  # [480*640, 3]

        # 将降维后的特征 reshape 为 [480, 640, 3]，并归一化到 [0, 255] 范围
        features_pca = features_pca.reshape(h, w, num_channels)
        features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min()) * 255
        features_pca = features_pca.astype(np.uint8)
    else:
        features_pca= torch.mean(feature_map,dim=0,keepdim=True)  # [H, W, C]
        features_pca= features_pca.permute(1,2,0).numpy()  # [H, W, C]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(features_pca, cmap=cmap)  # afmhot_r , binary , turbo , jet
    # plt.colorbar(im,label="Value")
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def tensor2img(tensor, min_max=(-1, 1), out_type=np.uint8):
    '''
    Converts a torch Tensor into an image Numpy array.
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order.
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default).
    '''
    tensor = tensor.detach().squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(n_img / 3), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def transform_tensor_imageNumpy(d):
    """
    d: 3D tensor(c,h,w)
    return: 3D numpy(h,w,c)
    """
    if d.min()==d.max():
        img_np= np.full(d.shape, d.min())
    else:
        d = (d - d.min()) / (d.max() - d.min())
        img_np = d.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    return img_np


def image_show(data):
    """
    data: 3D tensor(3,h,w)
    """
    data = transform_tensor_imageNumpy(data)
    w, h, c = data.shape
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    ax.imshow(data)  # afmhot_r , binary , turbo , jet
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


def tensor_show(data, cmap='turbo'):
    """
    data: 3D tensor(1,h,w)
    """
    data = transform_tensor_imageNumpy(data)
    w, h, c = data.shape
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    if cmap == 'gray':
        ax.imshow(data, cmap=cmap,vmin=0, vmax=1)
    else:
        ax.imshow(data, cmap=cmap)  # afmhot_r , binary , turbo , jet
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()


def visualization_tensor(tensor, batch_dim_exit=False, is_infrared=False,cmap='turbo',reduced_dim=1,show_info=True,save_path=None,use_pca=True,):
    """
    size: "b3hw" "bhw3" "3hw" "hw3" "bhw" "hw"
    """
    # assert shape in ["b3hw", "bhw3", "3hw", "hw3", "bhw", "hw","bchw", "bhwc", "chw","hwc"], "wzw :shape should be in ['b3hw', 'bhw3', '3hw', 'hw3', 'bhw', 'hw']"
    # if shape in ["bchw", "bhwc", "chw","hwc"]:
    #     assert num_channels in [1,3],"wzw: num_channels should be 3 or 1"
    tensor = tensor.detach().cpu()
    if len(tensor.shape)==3 and batch_dim_exit:
        tensor=tensor.unsqueeze(1)

    shape = tensor.shape
    if len(shape)==4:
        _,a,_,b=shape
        if a not in [1,3] and b not in [1,3]:
            print("-----注意特征的形状是[B,C,H,W]，现在的形状是{}-----".format(shape))
            for t in tensor:
                if show_info:
                    print(f"min: {t.min()}, max: {t.max()}, mean: {t.mean()}")
                show_feature_map(t, num_channels=reduced_dim, cmap=cmap,show_info=show_info,save_path=save_path,use_pca=use_pca)
        if b in [1,3]:
            tensor=tensor.permute(0,3,1,2)
            a=b

        if a==3:
            for t in tensor:
                if show_info:
                    print(f"min: {t.min()}, max: {t.max()}, mean: {t.mean()}")
                image_show(t)
        elif a==1:
            if is_infrared:
                for t in tensor:
                    if show_info:
                        print(f"min: {t.min()}, max: {t.max()}, mean: {t.mean()}")
                    image_show(t.repeat(3,1,1))
            else:
                for t in tensor:
                    if show_info:
                        print(f"min: {t.min()}, max: {t.max()}, mean: {t.mean()}")
                    tensor_show(t,cmap=cmap)
    elif len(shape)==3:
        a,_,b=shape

        if a not in [1,3] and b not in [1,3]:
            if show_info:
                print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
            print("-----wzw: 注意特征的形状是[C,H,W]，现在的形状是{}-----".format(shape))
            show_feature_map(tensor, num_channels=reduced_dim, cmap=cmap)
        if b in [1,3]:
            tensor=tensor.permute(2,0,1)
            a=b
        if a==3:
            if show_info:
                print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
            image_show(tensor)
        elif a==1:
            if is_infrared:
                if show_info:
                    print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
                image_show(tensor.repeat(3,1,1))
            else:
                if show_info:
                    print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
                tensor_show(tensor,cmap=cmap)

    elif len(shape)==2:
        if is_infrared:
            if show_info:
                print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
            tensor=tensor.unsqueeze(0).repeat(3,1,1)
            image_show(tensor)
        else:
            if show_info:
                print(f"min: {tensor.min()}, max: {tensor.max()}, mean: {tensor.mean()}")
            tensor_show(tensor.unsqueeze(0),cmap=cmap)

def visualization_token(tensor,cmap='turbo',type='BHTC',mean_head=True, h_w_ratio=0.75,show_info=False,save_path=None):
    shape=tensor.shape
    print('-----注意形状是，现在token的形状是{}-----'.format(shape))
    tensor = tensor.detach().cpu()
    def BLC(data,save_path=None):
        # batch,length,channel
        b,l,c=data.shape
        w = int(math.sqrt(int(data.shape[1] / h_w_ratio)))
        h= int(h_w_ratio*w)
        visualization_tensor(data.permute(0,2,1).view(b,-1,h,w),cmap=cmap,show_info=show_info,save_path=save_path)
    def BHTT(data,mean_head=False):
        # batch,head,token1_num,token2_num
        b,head,token1_num,token2_num=data.shape
        w = int(math.sqrt(int(token1_num / h_w_ratio)))
        h = int(h_w_ratio * w)
        if mean_head:
            a_map=data.mean(dim=1,keepdim=False)
            for t2 in range(token2_num):
                visualization_tensor(a_map[:,:,t2].view(b,h,w),cmap=cmap,batch_dim_exit=True,show_info=show_info)
        else:
            for h1 in range(head):
                a_map=data[:,h1,:,:]
                for t2 in range(token2_num):
                    visualization_tensor(a_map[:, :, t2].view(b, h, w), cmap=cmap,batch_dim_exit=True, show_info=show_info)
    def BHTC(data,mean_head=False):
        # batch,head,token_num,channel
        b,head,token_num,c=data.shape
        w = int(math.sqrt(int(token_num / h_w_ratio)))
        h = int(h_w_ratio * w)
        if mean_head:
            a_map=data.mean(dim=1,keepdim=False)
            visualization_tensor(a_map.permute(0,2,1).view(b,-1,h,w),cmap=cmap,batch_dim_exit=True,show_info=show_info)
        else:
            for h1 in range(head):
                a_map=data[:,h1,:,:]
                visualization_tensor(a_map.permute(0,2,1).view(b,-1,h,w),cmap=cmap,batch_dim_exit=True,show_info=show_info)
    if len(shape)==3:
        BLC(tensor,save_path=save_path)
    elif len(shape)==4:
        if type=='BHTT':
            BHTT(tensor,mean_head=mean_head)
        else:
            BHTC(tensor,mean_head=mean_head)

def auto_visualize_tensor(
    tensor,
    batch_index=0,
    reduced_dim=1,
    cmap='turbo',
    use_pca=True,
    show_info=False,
    is_image=None,
):
    """
    自动识别并可视化图像或特征图
    - 当 is_image 为 None，默认通道为3时视为图像，其他情况尝试自动判断
    - 如果通道为3但你知道它是特征图，请显式传 is_image=False
    """
    def v_2d(t,is_image=True):
        if is_image:
            image_show(tensor)

    def v_3d(t):
        if t.ndim == 2:
            return t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] in [1, 3]:
                return t
            elif t.shape[-1] in [1, 3]:
                return t.permute(2, 0, 1)
        raise ValueError(f"Unsupported shape {t.shape} for visualization")

    tensor = tensor.detach().cpu()

    # Batch 维度
    if tensor.ndim == 4:
        if batch_index >= tensor.shape[0]:
            print(f"⚠️ 输入 batch_index={batch_index} 超出范围，仅有 {tensor.shape[0]} 个样本")
            return
        tensor = tensor[batch_index]
        print(f"→ 显示 batch 中第 {batch_index} 个样本")

    # 统一为 [C, H, W]
    tensor = ensure_3d(tensor)
    c, h, w = tensor.shape

    # === 自动/手动判断是否为图像 ===
    if is_image is not None:
        is_img = is_image
    else:
        # 启发式判断：通道为3时视为图像；通道为1时看 infrared；其他视为特征图
        if c == 3:
            is_img = True
        elif c == 1:
            is_img = is_image
        else:
            is_img = False

    if is_img:
        print("→ 判断为图像")
        if c == 1:
            image_show(tensor.repeat(3,1,1))
        else:
            image_show(tensor)
    else:
        print("→ 判断为特征图")
        show_feature_map(tensor, num_channels=reduced_dim, cmap=cmap, show_info=show_info, use_pca=use_pca)




def visualization_attention_map_on_Image(image_tensor, attention_map, token_index, min_max=(-1, 1)):
    attention_token = attention_map[token_index].cpu().detach().numpy()
    attention_token[33] = 1
    attention_token[55] = 1
    attention_token[11] = 1
    attention_token[77] = 1
    attention_token[78] = 1
    attention_token[79] = 1
    attention_token = attention_token.reshape(16, 16)

    # 将图像 tensor 转换为 PIL 图像
    image_tensor = image_tensor.permute(1, 2, 0).numpy()  # 转换为 HWC 形状
    image_tensor = (image_tensor - min_max[0]) / (min_max[1] - min_max[0])
    image = Image.fromarray((image_tensor * 255).astype(np.uint8))  # 假设图像在 [0, 1] 范围，将其转换为 [0, 255]

    # 将注意力图缩放到与原图大小相同 (512x512)
    attention_image = Image.fromarray((attention_token * 255).astype(np.uint8))  # 归一化注意力图
    attention_image = attention_image.resize((512, 512))

    # 将注意力图转换为透明度图像（RGBA）进行叠加
    attention_image = attention_image.convert("RGBA")
    image = image.convert("RGBA")

    # 使用 alpha_composite 将注意力图叠加到原图上
    final_image = Image.alpha_composite(image, attention_image)

    # 显示结果
    plt.figure(figsize=(8, 8))
    plt.imshow(final_image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def visualize_top_patches(image, text_features, image_features, patch_size=(14, 14),text_token=None):
    """
    计算文本与图像特征的相似度，并将最相似的 patch 在原图像上标出。
    :param image: 原始图像 (3, 224, 224) tensor 格式
    :param text_features: 文本特征, 形状 (1, 77, 768)
    :param image_features: 图像特征, 形状 (1, 256, 768)
    :param image_size: 图像的 patch 网格尺寸，例如 (14, 14) 表示 14x14 的 patch
    :param min_max: 图像像素值范围，例如 (-1, 1)
    """
    image =image.detach().cpu()
    c,h,w=image.shape
    p_h,p_w=patch_size
    image = (image - image.min()) / (image.max() - image.min())
    # 计算相似度 (1, 77, 256)
    similarity = torch.matmul(text_features, image_features.transpose(-1, -2))

    # 获取每个文本 token 相似度最高的 patch 索引 (1, 77)
    top_patches = similarity.argmax(dim=-1).squeeze(0)
    if text_token is not None:
        tokens = text_token['input_ids'][0]  # 获取 tokens
        exclude_values = torch.tensor([49406, 49407, 267]).to(tokens.device)  # 排除的值
        # 生成布尔掩码，表示哪些 tokens 不在 exclude_values 中
        mask = ~torch.isin(tokens, exclude_values)
        # 使用掩码过滤 top_patches
        top_patches = top_patches[mask]

    # 获取 patch 坐标
    num_h, num_w = h // p_h, w // p_w

    # 转换图像格式为 (H, W, C) 以便显示
    image_np = image.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    ax.imshow(image_np)
    for i, patch_idx in enumerate(top_patches):
        y, x = divmod(patch_idx.item(),num_w)  # 计算 patch 在网格中的位置
        rect = plt.Rectangle((x * p_w, y * p_h), p_w, p_h,
                             linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

if __name__ == '__main__':
    vi1=Image.open("../TestData/vi/00004N.png")
    ir1=Image.open("../TestData/ir/00004N.png")
    vi2=Image.open("../TestData/vi/00008N.png")
    ir2=Image.open("../TestData/ir/00008N.png")
    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    vi1 = image_transforms(vi1)
    ir1 = image_transforms(ir1)
    tensor_show(ir1,cmap='gray')
    vi2 = image_transforms(vi2)
    ir2 = image_transforms(ir2)
    # 测试4维
    ## BCHW
    bchw=torch.concat()

