import os
import argparse
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, AutoTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from modules import TPBNet
from modules import PromptFusionModule
from modules import TICrossModalFusionModule,VAEFeatureFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from visualize_feature import show_feature_map
from Myutils.visualize import visualization_image_tensor
from Myutils.visualize_feature import show_feature_map
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def visualize_top_patches(image, text_features, image_features, image_size=(14, 14),min_max=(-1,1)):
        """
        计算文本与图像特征的相似度，并将最相似的 patch 在原图像上标出。
        :param image: 原始图像 (3, 224, 224) tensor 格式
        :param text_features: 文本特征, 形状 (1, 77, 768)
        :param image_features: 图像特征, 形状 (1, 256, 768)
        :param image_size: 图像的 patch 网格尺寸，例如 (14, 14) 表示 14x14 的 patch
        :param min_max: 图像像素值范围，例如 (-1, 1)
        """
        image=(image-min_max[0])/(min_max[1]-min_max[0])
        # 计算相似度 (1, 77, 256)
        similarity = torch.matmul(text_features, image_features.transpose(-1, -2))

        # 获取每个文本 token 相似度最高的 patch 索引 (1, 77)
        top_patches = similarity.argmax(dim=-1).squeeze(0)

        # 获取 patch 坐标
        num_patches = image_size[0] * image_size[1]
        patch_h, patch_w = image.shape[1] // image_size[0], image.shape[2] // image_size[1]

        # 转换图像格式为 (H, W, C) 以便显示
        image_np = image.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots()
        ax.imshow(image_np)

        for i, patch_idx in enumerate(top_patches):
            y, x = divmod(patch_idx.item(), image_size[1])  # 计算 patch 在网格中的位置
            rect = plt.Rectangle((x * patch_w, y * patch_h), patch_w, patch_h,
                                 linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path",default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'],)
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, help="only set this to True for lowlight and highlight tasks")

    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/multiFusion_attnBeforeFuse-0304a/pick-checkpoint-54000/", required=False,)
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False,)
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--num_inference_steps", type=int, default=20,)
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--save_root", default="Test_Results/multiFusion_attnBeforeFuse-0304a/" )
    
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--img_path", type=str, default="./TestData")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_TPBNet_path = os.path.join(args.ckpt_dir, "img1_tpb_net.pt")
    img2_TPBNet_path = os.path.join(args.ckpt_dir, "img2_tpb_net.pt")
    vae_fuse_net_path = os.path.join(args.ckpt_dir, "vae_fuse_net.pt")
    tiCrossModal_fusion_net_path = os.path.join(args.ckpt_dir, "tiCrossModal_fusion_net.pt")
    print('--------loading img1_TPB from:  ', img1_TPBNet_path, 'img2_TPB from: ',img2_TPBNet_path, '----------------------')
    os.makedirs(args.save_root, exist_ok=True)

    
    # Step-2: instantiate models and schedulers
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=None).to(device)
    # unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=None).to(device)
    # text_encoder = CLIPTextModel.from_pretrained(args.clip_path).to(device)
    # clip_v = CLIPVisionModel.from_pretrained(args.clip_path).to(device)
    # noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    vae = AutoencoderKL.from_pretrained('./pretrained-large-modal/VAE', subfolder="vae", revision=None).to(device)
    unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None).to(device)
    text_encoder = CLIPTextModel.from_pretrained("./pretrained-large-modal/text_encoder").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("./pretrained-large-modal/tokenizer")
    image_encoder = CLIPVisionModel.from_pretrained("./pretrained-large-modal/clip_vision").to(device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")
    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)


    img1_tpb_net = TPBNet().to(device)
    img2_tpb_net = TPBNet().to(device)
    vae_fuse_net = VAEFeatureFusionModule().to(device)
    tiCrossModal_fusion_net = TICrossModalFusionModule(768).to(device)
    try:
        # tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
        img1_tpb_net.load_state_dict(torch.load(img1_TPBNet_path,weights_only=True)['model'], strict=True)
        img2_tpb_net.load_state_dict(torch.load(img2_TPBNet_path,weights_only=True)['model'], strict=True)
        vae_fuse_net.load_state_dict(torch.load(vae_fuse_net_path,weights_only=True)['model'], strict=True)
        tiCrossModal_fusion_net.load_state_dict(torch.load(tiCrossModal_fusion_net_path,weights_only=True)['model'], strict=True)
    except:
        img1_tpb_net = torch.nn.DataParallel(img1_tpb_net)
        img2_tpb_net = torch.nn.DataParallel(img2_tpb_net)
        vae_fuse_net = torch.nn.DataParallel(vae_fuse_net)
        # tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
        img1_tpb_net.load_state_dict(torch.load(img1_TPBNet_path, weights_only=True)['model'], strict=True)
        img2_tpb_net.load_state_dict(torch.load(img2_TPBNet_path, weights_only=True)['model'], strict=True)
        vae_fuse_net.load_state_dict(torch.load(vae_fuse_net_path, weights_only=True)['model'], strict=True)
        tiCrossModal_fusion_net.load_state_dict(torch.load(tiCrossModal_fusion_net_path, weights_only=True)['model'], strict=True)
 

    img1_tpb_net.eval()
    img2_tpb_net.eval()
    vae_fuse_net.eval()
    tiCrossModal_fusion_net.eval()



    # Step-3: prepare data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder=os.path.join(args.img_path,'img1')
    img2_image_folder=os.path.join(args.img_path,'img2')
    text_folder=os.path.join(args.img_path,'DiS-IF_test_text_detail_content')
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

    text_path=[
        os.path.join(text_folder,file)
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

    img1_images=[load_image(path) for path in img1_image_path]
    img2_images=[load_image(path) for path in img2_image_path]
    text_list=[open(path, 'r', encoding='utf-8').read() for path in text_path]


    

    with torch.no_grad():
        # TPB
        for index,(img1,img2,text) in enumerate(zip(img1_images,img2_images,text_list)):
            image_transforms = transforms.Compose(
                [
                    transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR),
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
            # img1_clip_visual_input = clip_image_processor(images=img1, return_tensors="pt").pixel_values.to(device=vae.device)
            # img2_clip_visual_input = clip_image_processor(images=img2, return_tensors="pt").pixel_values.to(device=vae.device)

            img1_prompt_guidance = img1_tpb_net(clip_vision_outputs=image_encoder(img1_normalized_pixel_values, output_attentions=True, output_hidden_states=True),
                                                use_global=args.used_clip_vision_global,
                                                layer_ids=args.used_clip_vision_layers, )
            img2_prompt_guidance = img2_tpb_net(clip_vision_outputs=image_encoder(img2_normalized_pixel_values, output_attentions=True, output_hidden_states=True),
                                                use_global=args.used_clip_vision_global,
                                                layer_ids=args.used_clip_vision_layers, )



            text = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).to(device=vae.device)

            text_features = text_encoder(**text, output_attentions=True, output_hidden_states=True).last_hidden_state
            fusion_prompt_guidance = tiCrossModal_fusion_net(img1_prompt_guidance,img2_prompt_guidance,text_features)
            # resolution adjustment (one can adjust this resolution also, as long as the short side is equal to or larger than 512)
            width, height = img1.size


            # pre-process image
            img1 = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)  # image now is tensor in [-1,1]
            img2 = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)  # image now is tensor in [-1,1]

            img1_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img1)), 2, dim=1)[0]
            img2_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img2)), 2, dim=1)[0]

            down_block_res_samples=vae_fuse_net(img1_scb_cond,img2_scb_cond)

            b, c, h, w = img1_scb_cond.size()

            # set/load random seed
            generator = torch.Generator()
            generator.manual_seed(args.seed) # one can also adjust this seed to get different results

            # set the noise or latents
            if args.inp_of_unet_is_random_noise:
                latents = torch.randn((1,4, h, w), generator=generator).cuda()
            else:
                noise = torch.randn((1,4, h, w), generator=generator).cuda()

            # set the time step
            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()
            # feedforward
            for i, t in enumerate(timesteps):
                # add noise
                if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                    latents = noise_scheduler.add_noise(img1_scb_cond, noise, t, )
                # diffusion unet
                noise_pred = unet(latents,
                    t,
                    encoder_hidden_states=fusion_prompt_guidance,
                    down_block_additional_residuals= down_block_res_samples,
                ).sample

                # update the latents
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # post-process
            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred = vae_image_processor.postprocess(pred, output_type='pil')[0]
    
            # save_= concat_imgs([pil_image.resize(pred.size), pred], target_size=pred.size, target_dim=1)
            save_= concat_imgs([pred], target_size=pred.size, target_dim=1)
            # save_.save(os.path.join('./temp_results/', os.path.basename(args.img_path)))
            save_.save(os.path.join(args.save_root,os.path.basename(img1_image_path[index])))
            print(f'--------{index} done-----------')
    print('--------all done-----------')
