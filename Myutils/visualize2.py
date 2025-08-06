import numpy as np
import matplotlib.pyplot as plt
import math

from torchvision.utils import make_grid
from PIL import Image
import torch

from sklearn.decomposition import PCA
from torchvision import transforms

def show_feature_map(feature_map, pca_num_channels=1, cmap='turbo', save_path=None,use_pca=True):
    """
    可视化特征图
    :param use_pca:
    :param save_path:
    :param cmap: 颜色映射
    :param feature_map: 特征图，[C, H, W]
    :param pca_num_channels: 可视化通道数
    """

    feature_map = feature_map.cpu()
    c, h, w = feature_map.shape
    # 使用 PCA 将 128 维降到 n 维
    if use_pca:
        features_reshaped = feature_map.view(c, -1).T.numpy()  # [480*640, 128]
        pca = PCA(n_components=pca_num_channels)
        features_pca = pca.fit_transform(features_reshaped)  # [480*640, 3]

        # 将降维后的特征 reshape 为 [480, 640, 3]，并归一化到 [0, 255] 范围
        features_pca = features_pca.reshape(h, w, pca_num_channels)
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


def image_show(data,save_path=None):
    """
    data: 3D tensor(3,h,w)
    """
    data = transform_tensor_imageNumpy(data)
    w, h, c = data.shape
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    ax.imshow(data)  # afmhot_r , binary , turbo , jet
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def d1_tensor_show(data, cmap='turbo',save_path=None):
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
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()




def visualize_token(tensor,cmap='turbo',type=None,mean_head=True, h_w_ratio=0.75,show_info=False,save_path=None,num_sample=None,num_token=None):
    shape=tensor.shape
    if show_info:
        for i in tensor:
            print('-----Sample: max: {}, min: {}-----'.format(i.max(), i.min()))
    if tensor.ndim==3:
        b, _, t = shape
    elif tensor.ndim==4:
        b, _, _, t,= shape

    type = 'BHTT' if type is None and t in [77, 256] else type
    num_sample=b if num_sample is None else num_sample
    tensor = tensor.detach().cpu()
    def BLC(data,num_sample,save_path=None):
        # batch,length,channel
        b,l,c=data.shape
        w = int(math.sqrt(int(data.shape[1] / h_w_ratio)))
        h= int(h_w_ratio*w)
        visualize_tensor(data.permute(0,2,1).view(b,-1,h,w),cmap=cmap,show_info=False,save_path=save_path,num_sample=num_sample)
    def BHTT(data,num_sample,mean_head=False,num_token=None):
        b,head,token1_num,token2_num=data.shape
        if num_token is None:
            num_token=token2_num
        w = int(math.sqrt(int(token1_num / h_w_ratio)))
        h = int(h_w_ratio * w)
        if mean_head:
            a_map=data.mean(dim=1,keepdim=False)
            for t2 in range(num_token):
                visualize_tensor(a_map[:,:,t2].view(b,h,w),cmap=cmap,batch_exist=True,show_info=False,num_sample=num_sample,is_image=False)
        else:
            for h1 in range(head):
                a_map=data[:,h1,:,:]
                for t2 in range(num_token):
                    visualize_tensor(a_map[:, :, t2].view(b, h, w), cmap=cmap,batch_exist=True, show_info=False,num_sample=num_sample)
    def BHTC(data,num_sample,mean_head=False):
        # batch,head,token_num,channel
        b,head,token_num,c=data.shape
        w = int(math.sqrt(int(token_num / h_w_ratio)))
        h = int(h_w_ratio * w)
        if mean_head:
            a_map=data.mean(dim=1,keepdim=False)
            visualize_tensor(a_map.permute(0,2,1).view(b,-1,h,w),cmap=cmap,batch_exist=True,show_info=False,num_sample=num_sample)
        else:
            for h1 in range(head):
                a_map=data[:,h1,:,:]
                visualize_tensor(a_map.permute(0,2,1).view(b,-1,h,w),cmap=cmap,batch_exist=True,show_info=False,num_sample=num_sample)

    if len(shape)==3:
        BLC(tensor,num_sample=num_sample,save_path=save_path)
    elif len(shape)==4:
        if type=='BHTT':
            BHTT(tensor,num_sample,mean_head=mean_head,num_token=num_token)
        else:
            BHTC(tensor,num_sample=num_sample, mean_head=mean_head)

def visualize_tensor(
    tensor,
    batch_exist=False,
    pca_num_channels=1,
    cmap='viridis',
    num_sample=None,
    use_pca=True,
    show_info=False,
    is_image=True,
    save_path=None,
):
    """
    自动识别并可视化图像或特征图
    - 当 is_image 为 None，默认通道为3时视为图像，其他情况尝试自动判断
    - 如果通道为3但你知道它是特征图，请显式传 is_image=False
    """
    tensor=tensor.detach().cpu()
    def s_v2(t):
        print('-----Sample: max: {}, min: {}-----'.format(t.max(), t.min()))
    def s_v3(t,batch_exist=False):
        if batch_exist:
            for i in t:
                s_v2(i)
        else:
            s_v2(t)
    def s_v4(t,batch_exist=False):
        for i in t:
            s_v3(i, batch_exist=batch_exist)

    if show_info:
        if tensor.ndim==3:
            s_v3(tensor, batch_exist=batch_exist)
        if tensor.ndim==4:
            s_v4(tensor, batch_exist=batch_exist)
    # 规范通道,将通道数放在高宽前面
    ndim=tensor.ndim
    if ndim == 3:
        a,_,b=tensor.shape
        if a not in [1,3] and b in [1,3]:
            tensor=tensor.permute(2,0,1)
    elif ndim == 4:
        _,a,_,b=tensor.shape
        if a not in [1,3] and b in [1,3]:
            tensor=tensor.permute(0,3,1,2)

    # 计算要显示的样本数
    if num_sample is None and ((ndim==3 and batch_exist) or ndim==4):
        num_sample = tensor.shape[0]



    def v_2d(t,is_image=True,pca_num_channels=1,cmap='viridis',save_path=None,use_pca=True):
        if is_image:
            d1_tensor_show(t.unsqueeze(0),cmap='gray',save_path=save_path)
        else:
            show_feature_map(t.unsqueeze(0),pca_num_channels=pca_num_channels, cmap=cmap,save_path=save_path,use_pca=use_pca)

    def v_3d(t,is_image=False,pca_num_channels=1,cmap='viridis',save_path=None,use_pca=True):
        if batch_exist:
            for i in tensor[:num_sample]:
                v_2d(i,is_image=is_image,pca_num_channels=pca_num_channels,cmap=cmap,save_path=save_path,use_pca=use_pca)
        else:
            c,_,_=t.shape
            if c==1:
                v_2d(t[0],is_image=is_image,pca_num_channels=pca_num_channels,cmap=cmap,save_path=save_path,use_pca=use_pca)
            elif c==3:
                if is_image:
                    image_show(t,save_path=save_path)
                else:
                    show_feature_map(t, pca_num_channels=pca_num_channels, cmap=cmap, use_pca=use_pca, save_path=save_path)
            else:
                show_feature_map(t, pca_num_channels=pca_num_channels, cmap=cmap, use_pca=use_pca, save_path=save_path)

    def v_4d(t,is_image=False,pca_num_channels=1,cmap='viridis',save_path=None,use_pca=True):
        for i in t[:num_sample,:,:,:]:
            v_3d(i,is_image=is_image,pca_num_channels=pca_num_channels,cmap=cmap,save_path=save_path,use_pca=use_pca)

    if ndim == 4:
        v_4d(tensor,is_image=is_image, pca_num_channels=pca_num_channels, cmap=cmap, save_path=save_path, use_pca=use_pca)
    elif ndim == 3:
        v_3d(tensor,is_image=is_image, pca_num_channels=pca_num_channels, cmap=cmap,  save_path=save_path, use_pca=use_pca)
    elif ndim==2:
        v_2d(tensor,is_image=is_image, pca_num_channels=pca_num_channels, cmap=cmap, save_path=save_path, use_pca=use_pca)




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
    vi1=Image.open(r"D:\Projects\PYTHON\Diff-Plugin-main\TrainData\VIFusion\MSRS\vi\00001D.png")
    ir1=Image.open(r"D:\Projects\PYTHON\Diff-Plugin-main\TrainData\VIFusion\MSRS\ir\00001D.png")
    vi2=Image.open(r"D:\Projects\PYTHON\Diff-Plugin-main\TrainData\VIFusion\MSRS\vi\00002D.png")
    ir2=Image.open(r"D:\Projects\PYTHON\Diff-Plugin-main\TrainData\VIFusion\MSRS\ir\00002D.png")
    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


    vi1 = image_transforms(vi1)
    ir1 = image_transforms(ir1)
    vi2 = image_transforms(vi2)
    ir2 = image_transforms(ir2)
    # 测试2维
    hw=ir2[0]
    # auto_visualize_tensor(hw,is_image=False,cmap='viridis')
    # auto_visualize_tensor(hw,cmap='viridis')

    # 测试3维
    ohw=ir1
    # auto_visualize_tensor(ohw)
    # auto_visualize_tensor(ohw,is_image=False)
    hwo=ir1.permute(1,2,0)
    # auto_visualize_tensor(hwo)
    # auto_visualize_tensor(hwo, is_image=False)
    # auto_visualize_tensor(hwo,batch_exist=True)
    # auto_visualize_tensor(hwo,batch_exist=True, is_image=False)
    thw=vi1
    # auto_visualize_tensor(thw)
    # auto_visualize_tensor(thw,is_image=False)
    # auto_visualize_tensor(thw,batch_exist=True)
    # auto_visualize_tensor(thw,batch_exist=True, is_image=False)
    chw=torch.cat([vi1,ir1],dim=0)
