import os
import argparse
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from torchvision import transforms
from diffusers.models.transformer_2d import Transformer2DModel
from modules import SCBNet
from modules import TPBNet
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from modules import TaskPromptFusionNet
from modules import FeatureFusionModule,Text2ImagePromptFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from visualize_feature import show_feature_map
from Myutils.visualize import visualization_tensor
import json
import torch.nn.functional as F
#=====================================================================
from diffusers.models.attention import Attention
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D
from modules import CustomEncoder,CustomDecoder,VAE_ShallowFeatureFusionModule, VAEFuseNet
from MyHook import *

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")
    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/multiFusion-0303a/pick-checkpoint-54000/", required=False, )
    parser.add_argument("--VAE_ckpt_dir", type=str, default="results/Stage_two/whole_network/pick-checkpoint-54000/", required=False,)
    parser.add_argument("--img_path", type=str, default="./TestData")
    parser.add_argument("--text_type", type=str, default="DiS-IF_test_text_detail_content")
    parser.add_argument("--save_root", default="Test_Results/Stage_two/54000-stpe5")
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--num_inference_steps", type=int, default=5, )


    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path",default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'],)
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, help="only set this to True for lowlight and highlight tasks")

    parser.add_argument("--used_clip_vision_global", action="store_true", default=False,)
    parser.add_argument("--resolution", type=int, default=512,)
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
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)

    features = {"attn_output": [], "attn_map": []}



    vae = AutoencoderKL.from_pretrained('./pretrained-large-modal/VAE', subfolder="vae", revision=None).to(device)
    vae.eval()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
    with open("./pretrained-large-modal/VAE/config.json", "r") as f:
        config = json.load(f)
    encoder=CustomEncoder(config,vae.encoder).to(device)
    encoder.load_state_dict(torch.load(os.path.join(args.VAE_ckpt_dir, "encoder.pt"),weights_only=True)['model'],strict=True)
    decoder=CustomDecoder(config,vae.decoder).to(device)
    decoder.load_state_dict(torch.load(os.path.join(args.VAE_ckpt_dir, "decoder.pt"),weights_only=True)['model'],strict=True)

    encoder.eval()
    decoder.eval()

    image_encoder = CLIPVisionModel.from_pretrained("./pretrained-large-modal/clip_vision").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("./pretrained-large-modal/tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("./pretrained-large-modal/text_encoder").to(device)

    # my part---------------------
    img1_tpb_net = TPBNet().to(device)
    img2_tpb_net = TPBNet().to(device)
    img1_tpb_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "img1_tpb_net.pt"), weights_only=True)['model'], strict=True)
    img2_tpb_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "img2_tpb_net.pt"), weights_only=True)['model'], strict=True)
    img1_tpb_net.eval()
    img2_tpb_net.eval()

    task_prompt_fusion_net = TaskPromptFusionNet(single_input_dim=768, output_dim=768).to(device)
    task_prompt_fusion_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "task_prompt_fusion_net.pt"),weights_only=True)['model'],strict=True)

    vae_fuse_net = VAEFuseNet().to(device)
    vae_fuse_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "vae_fuse_net.pt"),weights_only=True)['model'],strict=True)
    vae_fuse_net.eval()

    text2image_prompt_fusion_net = Text2ImagePromptFusionModule(768).to(device)
    text2image_prompt_fusion_net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "text2image_prompt_fusion_net.pt"),weights_only=True)['model'],strict=True)
    text2image_prompt_fusion_net.eval()

    vae_shallow_feature_fusion_module = VAE_ShallowFeatureFusionModule([128, 128, 256, 512, 512])
    vae_shallow_feature_fusion_module.load_state_dict(torch.load(os.path.join(args.VAE_ckpt_dir, "feature_fusion_module.pt"), weights_only=True)['model'],
                                                      strict=True)
    vae_shallow_feature_fusion_module.eval()


    # Step-3: prepare data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder=os.path.join(args.img_path,'img1')
    img2_image_folder=os.path.join(args.img_path,'img2')
    text_folder=os.path.join(args.img_path,args.text_type)

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
    # target_filenames = {"00209D"}
    # img1_image_path = [p for p in img1_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # img2_image_path = [p for p in img2_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # text_path = [p for p in text_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # -----------------

    img1_images=[load_image(path) for path in img1_image_path]
    img2_images=[load_image(path) for path in img2_image_path]
    text_list = [open(path, 'r', encoding='utf-8').read() for path in text_path]

    # pil222_image = image222.copy()
    

    with torch.no_grad():
        # TPB
        for index,(img1,img2,text) in enumerate(zip(img1_images,img2_images,text_list)):
            # context branch-----------------------
            image_transforms = transforms.Compose(
                [
                    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            image_mean = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1)
            image_std = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1)
            data1 = image_transforms(img1).unsqueeze(0)
            data2 = image_transforms(img2).unsqueeze(0)
            img1_normalized_pixel_values = (data1 + 1.0) / 2.0
            img1_normalized_pixel_values = torch.nn.functional.interpolate(img1_normalized_pixel_values, size=(224, 224), mode="bilinear",
                                                                           align_corners=False)
            img1_normalized_pixel_values = ((img1_normalized_pixel_values - image_mean) / image_std).to(device)

            img2_normalized_pixel_values = (data2 + 1.0) / 2.0
            img2_normalized_pixel_values = torch.nn.functional.interpolate(img2_normalized_pixel_values, size=(224, 224), mode="bilinear",
                                                                           align_corners=False)
            img2_normalized_pixel_values = ((img2_normalized_pixel_values - image_mean) / image_std).to(device)

            # clip_vision_outputs_1 = image_encoder(img1_normalized_pixel_values, output_attentions=True, output_hidden_states=True).hidden_states[
            #                            args.used_clip_vision_layers][
            #                       :, 1:, :]
            clip_vision_outputs_2 = image_encoder(img2_normalized_pixel_values, output_attentions=True, output_hidden_states=True).hidden_states[
                                        args.used_clip_vision_layers][
                                    :, 1:, :]
            # visualization_tensor(clip_vision_outputs_1.permute(0, 2, 1).view(1, -1, 16, 16))
            # visualization_tensor(clip_vision_outputs_2.permute(0, 2, 1).view(1, -1, 16, 16))

            img1_prompt_guidance = img1_tpb_net(
                clip_vision_outputs=image_encoder(img1_normalized_pixel_values, output_attentions=True, output_hidden_states=True),
                use_global=args.used_clip_vision_global,
                layer_ids=args.used_clip_vision_layers, )

            img2_prompt_guidance = img2_tpb_net(
                clip_vision_outputs=image_encoder(img2_normalized_pixel_values, output_attentions=True, output_hidden_states=True),
                use_global=args.used_clip_vision_global,
                layer_ids=args.used_clip_vision_layers, )

            fusion_prompt_guidance = task_prompt_fusion_net(img1_prompt_guidance, img2_prompt_guidance)

            text_token = tokenizer(text, return_tensors="pt", padding=False, truncation=True).to(device=vae.device)
            text_features = text_encoder(**text_token, output_attentions=True, output_hidden_states=True).last_hidden_state
            fusion_prompt_guidance, attn = text2image_prompt_fusion_net(fusion_prompt_guidance, text_features)
            fusion_prompt_guidance=torch.randn(size=fusion_prompt_guidance.size()).to(device)

            # visual branch----------------------
            width, height = img1.size
            img1_preprocess = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)
            z1,feature1_list = encoder(img1_preprocess, return_feature=True)
            z2,feature2_list = encoder(img2_preprocess, return_feature=True)

            img1_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z1), 2, dim=1)[0]
            img2_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z2), 2, dim=1)[0]

            down_block_res_samples = vae_fuse_net(img1_scb_cond, img2_scb_cond)

            b, c, h, w = img1_scb_cond.size()

            # set/load random seed
            generator = torch.Generator()
            generator.manual_seed(args.seed)  # one can also adjust this seed to get different results

            # set the noise or latents
            if args.inp_of_unet_is_random_noise:
                latents = torch.randn((1, 4, h, w), generator=generator).cuda()
            else:
                noise = torch.randn((1, 4, h, w), generator=generator).cuda()

            # set the time step
            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()
            # feedforward
            for i, t in enumerate(timesteps):
                # add noise
                # if i==19:
                #     for block in unet.down_blocks:
                #         if isinstance(block, CrossAttnDownBlock2D):
                #             for m in block.attentions:
                #                 if isinstance(m,Transformer2DModel):
                #                     m.register_forward_hook(Transformer2D_hook_fn, with_kwargs=True)
                if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                    latents = noise_scheduler.add_noise(img1_scb_cond, noise, t, )
                # diffusion unet
                noise_pred = unet(latents,
                                  t,
                                  encoder_hidden_states=fusion_prompt_guidance,
                                  down_block_additional_residuals=down_block_res_samples,
                                  ).sample

                # update the latents
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            feature_list = vae_shallow_feature_fusion_module(feature1_list, feature2_list)
            feature_list = [feature / vae.config.scaling_factor for feature in feature_list]

            rec = decoder(vae.post_quant_conv(latents / vae.config.scaling_factor), feature_list)
            rec = vae_image_processor.postprocess(rec, output_type='pil')[0]

            save_= concat_imgs([rec], target_size=rec.size, target_dim=1)
            save_.save(os.path.join(args.save_root,os.path.basename(img1_image_path[index])))
            print(f'--------{index} done-----------')
    print('--------all done-----------')
