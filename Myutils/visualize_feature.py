import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

def show_feature_map(feature_map, num_channels=1, save_path=None):
    """
    可视化特征图
    :param feature_map: 特征图，[C, H, W]
    :param num_channels: 可视化通道数
    """
    if feature_map.dim() == 4:
        feature_map = feature_map.squeeze(0)
        print("only visualizing the first image in the batch")
    assert feature_map.dim() == 3, "Feature map should be [C, H, W]"

    feature_map = feature_map.cpu()
    c, w, h = feature_map.shape
    features_reshaped = feature_map.view(c, -1).T.numpy()  # [480*640, 128]
    # 使用 PCA 将 128 维降到 n 维
    pca = PCA(n_components=num_channels)
    features_pca = pca.fit_transform(features_reshaped)  # [480*640, 3]

    # 将降维后的特征 reshape 为 [480, 640, 3]，并归一化到 [0, 255] 范围
    features_pca = features_pca.reshape(w, h, num_channels)
    features_pca = (features_pca - features_pca.min()) / (features_pca.max() - features_pca.min()) * 255
    features_pca = features_pca.astype(np.uint8)
    # 转换为 PIL 图像并显示
    # if features_pca.shape[2] == 1:
    #     features_pca = np.repeat(features_pca, 3, axis=2)  # 结果为 shape (480, 640, 3)
    # image = Image.fromarray(features_pca)
    # image.show()
    fig, ax = plt.subplots(figsize=(h / 100, w / 100))
    ax.imshow(features_pca, cmap='turbo')  # afmhot_r , binary , turbo , jet
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()





if __name__ == '__main__':
    # 加载图片特征
    pass
