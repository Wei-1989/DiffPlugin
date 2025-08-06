import os
import argparse
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer,CLIPModel
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import TPBNet
from modules import PromptFusionModule
from modules import Text2ImagePromptFusionModule, VAEFeatureFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from visualize_feature import show_feature_map
from Myutils.visualize import visualization_tensor, visualize_top_patches
from Myutils.visualize_feature import show_feature_map
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import open_clip



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")
    parser.add_argument("--img_path", type=str, default="./TestData/")
    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/multiFusion-0303a/pick-checkpoint-54000/", required=False, )
    parser.add_argument("--save_root", default="Test_Results/temp/")
    parser.add_argument("--text_type", default="PSFusion_test_categories")
    parser.add_argument("--used_clip_vision_layers", type=int, default=0, )

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'], )
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False,
                        help="only set this to True for lowlight and highlight tasks")

    parser.add_argument("--used_clip_vision_global", action="store_true", default=False, )
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--num_inference_steps", type=int, default=20, )
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')

    parser.add_argument("--seed", type=int, default=42, )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_root, exist_ok=True)

    model, preprocess,_ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')




    # Step-3: prepare data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(args.img_path, 'img1')
    img2_image_folder = os.path.join(args.img_path, 'img2')
    text_folder = os.path.join(args.img_path, args.text_type)
    img1_image_path = [
        os.path.join(img1_image_folder, file)
        for file in os.listdir(img1_image_folder)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]

    img2_image_path = [
        os.path.join(img2_image_folder, file)
        for file in os.listdir(img2_image_folder)
        if os.path.splitext(file)[1].lower() in image_extensions
    ]

    text_path = [
        os.path.join(text_folder, file)
        for file in os.listdir(text_folder)
        if os.path.splitext(file)[1].lower() in ['.txt']
    ]

    img1_image_path.sort(key=lambda x: os.path.basename(x))
    img2_image_path.sort(key=lambda x: os.path.basename(x))
    text_path.sort(key=lambda x: os.path.basename(x))

    # 只要特定文件-----------------
    # target_filenames = {"01458D","00283D","00055D"}
    # img1_image_path = [p for p in img1_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # img2_image_path = [p for p in img2_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # text_path = [p for p in text_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # -----------------

    img1_images = [load_image(path) for path in img1_image_path]
    img2_images = [load_image(path) for path in img2_image_path]
    text_list = [open(path, 'r', encoding='utf-8').read() for path in text_path]

    with torch.no_grad():
        # TPB
        for index, (img1, img2, text) in enumerate(zip(img1_images, img2_images, text_list)):
            print(text)
            image = preprocess(img1).unsqueeze(0)
            visualization_tensor(image)
            text = tokenizer(text)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            image = image.to(device)
            text = text.to(device)

            # 计算图像和文本的特征
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

            # 归一化特征
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 计算余弦相似度
            similarity = (image_features @ text_features.T)  # [1, N] 形状

            # 输出相似度
            print(similarity)

    print('--------all done-----------')
