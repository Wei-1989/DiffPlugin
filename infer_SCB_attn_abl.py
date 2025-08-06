import os
import argparse

import time

import numpy as np
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from torchvision import transforms
from diffusers.models.transformer_2d import Transformer2DModel
from modules import SCBNet_abl_attn
from modules import TPBNet
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from modules import TaskPromptFusionNet
from modules import FeatureFusionModule, Text2ImagePromptFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from Myutils.visualize import visualization_tensor
import json
import torch.nn.functional as F
# =====================================================================
from diffusers.models.attention import Attention
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D
from modules import CustomEncoder, CustomDecoder, VAE_ShallowFeatureFusionModule, VAEFuseNet, TextAdapter, MyUnet
from MyHook import *
import torch.nn as nn
from pick_filename import pick_file_name,token_name


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")
    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/Abl_attn/checkpoint-6000/", required=False, )
    parser.add_argument("--VAE_ckpt_dir", type=str, default="results/Stage_two/supervised_DiS/pick-checkpoint-54000/", required=False, )
    parser.add_argument("--img_path", type=str, default="./TestData/")
    parser.add_argument("--text_type", type=str, default="gt_category_no_commas")
    parser.add_argument("--save_root", default="Test_Results/abl_000/Fusion results")
    parser.add_argument("--used_clip_vision_layers", type=int, default=24, )
    parser.add_argument("--num_inference_steps", type=int, default=5, )
    parser.add_argument("--use_vae_refine", action="store_true", default=False, )

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

    # Step-2: instantiate models and schedulers
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to(device)
    # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to(device)
    # text_encoder = CLIPTextModel.from_pretrained(args.clip_path).to(device)
    # clip_v = CLIPVisionModel.from_pretrained(args.clip_path).to(device)
    # noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # pretrain part---------------------
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")
    # unet=UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)
    # unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)
    #  myunet------------------------------------
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        "./pretrained-large-modal/unet",
        subfolder="unet",
        revision=None,
    )

    # 2. 初始化你的自定义模型
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
    unet.fusion_weight.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "unet_fusion_weight.pt"), weights_only=True)['model'], strict=True)
    unet.eval()
    del pretrained_unet

    features = {"attn_output": [], "attn_map": []}

    vae = AutoencoderKL.from_pretrained('./pretrained-large-modal/VAE', subfolder="vae", revision=None).to(device)
    vae.eval()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    with open("./pretrained-large-modal/VAE/config.json", "r") as f:
        config = json.load(f)
    encoder = CustomEncoder(config, vae.encoder).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.VAE_ckpt_dir, "encoder.pt"), weights_only=True)['model'], strict=True)
    decoder = CustomDecoder(config, vae.decoder).to(device)
    decoder.load_state_dict(torch.load(os.path.join(args.VAE_ckpt_dir, "decoder.pt"), weights_only=True)['model'], strict=True)

    encoder.eval()
    decoder.eval()

    image_encoder = CLIPVisionModel.from_pretrained("./pretrained-large-modal/clip_vision").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("./pretrained-large-modal/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("./pretrained-large-modal/text_encoder").to(device)

    # my part---------------------
    # img1_tpb_net = TPBNet(1,).to(device)
    # img2_tpb_net = TPBNet(1).to(device)
    # img1_tpb_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "img1_tpb_net.pt"), weights_only=True)['model'], strict=True)
    # img2_tpb_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "img2_tpb_net.pt"), weights_only=True)['model'], strict=True)
    # img1_tpb_net.eval()
    # img2_tpb_net.eval()

    # task_prompt_fusion_net = TaskPromptFusionNet(single_input_dim=768, output_dim=768).to(device)
    # task_prompt_fusion_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "task_prompt_fusion_net.pt"),weights_only=True)['model'],strict=True)

    # text2image_prompt_fusion_net = Text2ImagePromptFusionModule(768).to(device)
    # text2image_prompt_fusion_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "text2image_prompt_fusion_net.pt"),weights_only=True)['model'],strict=True)
    # text2image_prompt_fusion_net.eval()

    vae_shallow_feature_fusion_module = VAE_ShallowFeatureFusionModule().to(device)
    vae_shallow_feature_fusion_module.load_state_dict(
        torch.load(os.path.join(args.VAE_ckpt_dir, "feature_fusion_module.pt"), weights_only=True)['model'], strict=True)
    vae_shallow_feature_fusion_module.eval()
    #
    # textAdapter=TextAdapter(2,768).to(device)
    # textAdapter.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "textAdapter.pt"), weights_only=True)['model'], strict=True)
    # textAdapter.eval()

    backup_unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None)
    if type(args.down_block_types) != list:
        args.down_block_types = [args.down_block_types]
    if type(args.block_out_channels) != list:
        args.block_out_channels = [args.block_out_channels]
    backup_unet.config.down_block_types = args.down_block_types
    backup_unet.config.block_out_channels = args.block_out_channels
    backup_unet.config.in_channels = 4

    scb_net = SCBNet_abl_attn.from_unet(backup_unet, load_weights_from_unet=False).to(device)
    scb_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "scb_net.pt"), weights_only=True)['model'], strict=True)
    scb_net.eval()
    del backup_unet

    # Step-3: prepare data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(args.img_path, 'vi')
    img2_image_folder = os.path.join(args.img_path, 'ir')
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
    # target_filenames = pick_file_name  # "00123D","00196D","01458D"
    # img1_image_path = [p for p in img1_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # img2_image_path = [p for p in img2_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # text_path = [p for p in text_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # -----------------

    img1_images = [load_image(path) for path in img1_image_path]
    img2_images = [load_image(path) for path in img2_image_path]
    text_list = [open(path, 'r', encoding='utf-8').read() for path in text_path]

    # pil222_image = image222.copy()

    with torch.no_grad():
        # TPB
        time_consume_list = []
        for index, (img1, img2, text) in enumerate(zip(img1_images, img2_images, text_list)):

            tokenizer.add_tokens(['bollard'])
            text_encoder.resize_token_embeddings(len(tokenizer))
            text_token = tokenizer(text, return_tensors="pt", padding=False, truncation=True).to(device=vae.device)
            text_features = text_encoder(**text_token, output_attentions=True, output_hidden_states=True).last_hidden_state

            # visual branch----------------------
            width, height = img1.size
            img1_preprocess = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)

            z1, feature1_list = encoder(img1_preprocess, return_feature=True)
            z2, feature2_list = encoder(img2_preprocess, return_feature=True)

            img1_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z1), 2, dim=1)[0]
            img2_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z2), 2, dim=1)[0]

            # for (name1, param1), (name2, param2) in zip(temp_net.down_blocks.named_parameters(), scb_net.down_blocks.named_parameters()):
            #     if not torch.equal(param1, param2):
            #         print(f"❌ 参数不同: {name1} vs {name2}")
            #     else:
            #         print(f"✅ 参数相同: {name1} vs {name2}")

            b, c, h, w = img1_scb_cond.size()

            # set/load random seed
            generator = torch.Generator()
            generator.manual_seed(args.seed)  # one can also adjust this seed to get different results

            # set the noise or latents
            # if args.inp_of_unet_is_random_noise:
            latents = torch.randn((1, 4, h, w), generator=generator).cuda()
            # else:
            #     latents = torch.randn((1, 4, h, w), generator=generator).cuda()

            # set the time step
            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()
            start_time = time.time()
            # feedforward
            for i, t in enumerate(timesteps):
                # add noise
                # if i == args.num_inference_steps-1:
                #     for name, module in unet.named_modules():
                #         if isinstance(module, CrossAttnDownBlock2D):
                #             module.register_forward_hook(CrossAttnDownBlock2D_hook_fn, with_kwargs=True)
                # if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                #     latents = noise_scheduler.add_noise(img1_scb_cond, noise, t, )
                # diffusion unet
                down_block_res_samples = scb_net(
                    timestep=t,
                    encoder_hidden_states=text_features,
                    cond_img=torch.concat([img1_scb_cond, img2_scb_cond], dim=1),
                    return_dict=False,
                    t_idx=9999,
                )

                # if i == args.num_inference_steps - 1:
                #     for d_feature in down_block_res_samples[:9]:
                #         visualization_tensor(d_feature)

                noise_pred = unet(latents,
                                  t,
                                  encoder_hidden_states=text_features,
                                  down_block_additional_residuals=down_block_res_samples,
                                  ).sample

                # update the latents
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # if i==len(timesteps)-1:
                #     torch.save(latents, os.path.join(args.save_root,os.path.basename(img1_image_path[index]).split('.')[0])+'_latents.pt')
            end_time = time.time()
            time_consume_list.append(end_time - start_time)

            feature_list = vae_shallow_feature_fusion_module(feature1_list, feature2_list)
            feature_list = [feature / vae.config.scaling_factor for feature in feature_list]
            if args.use_vae_refine:
                rec = decoder(vae.post_quant_conv(latents / vae.config.scaling_factor), feature_list)
            else:
                rec = decoder(vae.post_quant_conv(latents / vae.config.scaling_factor))
            rec = vae_image_processor.postprocess(rec, output_type='pil')[0]

            save_ = concat_imgs([rec], target_size=rec.size, target_dim=1)
            save_.save(os.path.join(args.save_root, os.path.basename(img1_image_path[index])))
            print(f'--------{index} done-----------')
        print('--------average time: {}-----------'.format(np.mean(time_consume_list)))
    print('--------all done-----------')
