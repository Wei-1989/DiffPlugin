import os
import sys

import torch

# 获取当前脚本的上一级目录作为项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
sys.path.insert(0, project_root)  # 如果你希望能 import 根目录下的模块
import argparse

import time

import numpy as np
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer,CLIPTokenizerFast
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler,PNDMScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from torchvision import transforms
from diffusers.models.transformer_2d import Transformer2DModel
from modules import SCBNet
from modules import TPBNet
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from modules import TaskPromptFusionNet
from modules import FeatureFusionModule, Text2ImagePromptFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from Myutils import visualization_tensor
import json
import torch.nn.functional as F
# =====================================================================
from diffusers.models.attention import Attention
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D
from original_SD.original_my_Unet2DConditionModel import My_Unet as MyUnet
from MyHook import *
import torch.nn as nn
from pick_filename import pick_file_name, token_name


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")
    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/MyUnet_0424a/pick-checkpoint-6500/", required=False, )
    parser.add_argument("--VAE_ckpt_dir", type=str, default="results/Stage_two/supervised_DiS/pick-checkpoint-54000/", required=False, )
    parser.add_argument("--img_path", type=str, default="./TestData")
    parser.add_argument("--text_type", type=str, default="gt_category_no_commas")
    parser.add_argument("--save_root", default="Test_Results/temp/Fusion results")
    parser.add_argument("--used_clip_vision_layers", type=int, default=24, )
    parser.add_argument("--num_inference_steps", type=int, default=50,)
    parser.add_argument("--use_vae_refine", action="store_true", default=True, )
    parser.add_argument("--guidance_scale", type=float, default=7.5 )

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path", default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'], )
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=True, help="only set this to True for lowlight and highlight "
                                                                                                 "tasks")
    parser.add_argument("--down_block_types", type=str, nargs="+",
                        default=["My_CrossAttnDownBlock2D", "My_CrossAttnDownBlock2D", "My_CrossAttnDownBlock2D", "DownBlock2D"], )
    parser.add_argument("--block_out_channels", type=int, nargs="+", default=[320, 640, 1280, 1280])

    parser.add_argument("--used_clip_vision_global", action="store_true", default=False, )
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--seed", type=int, default=42,)

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

    # Step-2: instantiate models and schedulers
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to(device)
    # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to(device)
    # text_encoder = CLIPTextModel.from_pretrained(args.clip_path).to(device)
    # clip_v = CLIPVisionModel.from_pretrained(args.clip_path).to(device)
    # noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # pretrain part---------------------
    # noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")
    noise_scheduler = PNDMScheduler.from_pretrained("./pretrained-large-modal/PNDMScheduler", subfolder="scheduler")
    # unet=UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)
    # unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)
    #  myunet------------------------------------
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        "./pretrained-large-modal/unet",
        subfolder="unet",
        revision=None,
    ).to(device)

    unet = MyUnet(**pretrained_unet.config).to(device)  # 使用相同配置

    # 3. 复制匹配的权重
    pretrained_state_dict = pretrained_unet.state_dict()
    my_state_dict = unet.state_dict()

    # 仅复制名称和形状匹配的参数
    for name, param in pretrained_state_dict.items():
        if name in my_state_dict and param.shape == my_state_dict[name].shape:
            my_state_dict[name].copy_(param)
        else:
            print(f"Skipping {name}: shape mismatch or not found in custom model.")
    # 5. 加载最终权重
    unet.load_state_dict(my_state_dict, strict=True)  # strict=False 允许部分加载
    unet.eval()
    del pretrained_unet


    features = {"attn_output": [], "attn_map": []}

    vae = AutoencoderKL.from_pretrained('./pretrained-large-modal/VAE', subfolder="vae", revision=None).to(device)
    vae.eval()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    with open("./pretrained-large-modal/VAE/config.json", "r") as f:
        config = json.load(f)


    tokenizer = CLIPTokenizerFast.from_pretrained("./pretrained-large-modal/tokenizer_fast")
    text_encoder = CLIPTextModel.from_pretrained("./pretrained-large-modal/text_encoder").to(device)



    with torch.no_grad():
        # TPB
        time_consume_list = []
        text='person, car, curve'

        text_token = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True).to(device=vae.device)
        text_features = text_encoder(text_token.input_ids)[0]


        uncond_input = tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).to(device)
        uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

        embeddings = torch.cat([uncond_embeddings, text_features])


        # set/load random seed
        generator = torch.Generator()
        generator.manual_seed(args.seed)  # one can also adjust this seed to get different results

        # set the noise or latents
        # if args.inp_of_unet_is_random_noise:
        latents = torch.randn((1, 4, 60, 80), generator=generator).cuda()

        # else:
        #     latents = torch.randn((1, 4, h, w), generator=generator).cuda()

        # set the time step
        noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
        timesteps = noise_scheduler.timesteps
        timesteps = timesteps.long()
        start_time = time.time()
        # feedforward
        for i, t in enumerate(timesteps):
            latent_input = torch.cat([latents]*2)
            encoder_input = torch.cat([uncond_embeddings, text_features])


            noise_pred = unet(latent_input,
                              t,
                              encoder_hidden_states=encoder_input,
                              ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 更新 latent
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        end_time = time.time()
        time_consume_list.append(end_time - start_time)

        rec = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        rec = vae_image_processor.postprocess(rec, output_type='pil')[0]
        save_ = concat_imgs([rec], target_size=rec.size, target_dim=1)
        save_.save(os.path.join(args.save_root, f'apple_{args.num_inference_steps}.png'))
    print('--------all done-----------')
