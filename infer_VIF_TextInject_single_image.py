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

from modules import SCBNet
from modules import TPBNet
from modules import PromptFusionModule
from modules import FeatureFusionModule,Text2ImagePromptFusionModule
from utils import concat_imgs, import_model_class_from_model_name_or_path
from visualize_feature import show_feature_map
from Myutils.visualize import visualization_image_tensor




def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin inference script.")

    parser.add_argument("--pretrained_model_name_or_path", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--clip_path",default="openai/clip-vit-large-patch14")
    parser.add_argument("--inp_of_crossatt", type=str, default='clip', choices=['text', 'clip'],)
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False, help="only set this to True for lowlight and highlight tasks")

    parser.add_argument("--ckpt_dir", type=str, default="results/VIFusion/WithText_refine_0224a/pick-checkpoint-25000/", required=False,)
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False,)
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--num_inference_steps", type=int, default=20,)
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--save_root", default="Test_Results/single_image/" )
    
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--img_path", type=str, default="./TestData")
    parser.add_argument("--file_name", type=str, default="00209D.png")
    parser.add_argument("--text_prompt", type=str, default="A man is standing besides a car.")


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_args()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SCBNet_path = os.path.join(args.ckpt_dir, "scb_net")
    img1_TPBNet_path = os.path.join(args.ckpt_dir, "img1_tpb_net.pt")
    img2_TPBNet_path = os.path.join(args.ckpt_dir, "img2_tpb_net.pt")
    task_prompt_fusion_net_path = os.path.join(args.ckpt_dir, "task_prompt_fusion_net.pt")
    text2image_prompt_fusion_net_path = os.path.join(args.ckpt_dir, "text2image_prompt_fusion_net.pt")
    feature_fusion_net_path = os.path.join(args.ckpt_dir, "feature_fusion_net.pt")
    print('--------loading SCB from: ', SCBNet_path, '   , img1_TPB from:  ', img1_TPBNet_path, 'img2_TPB from: ',img2_TPBNet_path, '----------------------')
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
    clip_v = CLIPVisionModel.from_pretrained("./pretrained-large-modal/clip_vision").to(device)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")
    clip_image_processor = CLIPImageProcessor()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)


    scb_net = SCBNet.from_pretrained(SCBNet_path).to(device)
    img1_tpb_net = TPBNet().to(device)
    img2_tpb_net = TPBNet().to(device)
    task_prompt_fusion_net=PromptFusionModule(single_input_dim=768, output_dim=768).to(device)
    text2image_prompt_fusion_net = Text2ImagePromptFusionModule(768).to(device)
    feature_fusion_net = FeatureFusionModule(4, 4).to(device)
    try:
        # tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
        img1_tpb_net.load_state_dict(torch.load(img1_TPBNet_path,weights_only=True)['model'], strict=True)
        img2_tpb_net.load_state_dict(torch.load(img2_TPBNet_path,weights_only=True)['model'], strict=True)
        task_prompt_fusion_net.load_state_dict(torch.load(task_prompt_fusion_net_path,weights_only=True)['model'], strict=True)
        text2image_prompt_fusion_net.load_state_dict(torch.load(text2image_prompt_fusion_net_path,weights_only=True)['model'], strict=True)
        feature_fusion_net.load_state_dict(torch.load(feature_fusion_net_path,weights_only=True)['model'], strict=True)
    except:
        img1_tpb_net = torch.nn.DataParallel(img1_tpb_net)
        img2_tpb_net = torch.nn.DataParallel(img2_tpb_net)
        task_prompt_fusion_net = torch.nn.DataParallel(task_prompt_fusion_net)
        feature_fusion_net = torch.nn.DataParallel(feature_fusion_net)
        # tpb_net.load_state_dict(torch.load(TPBNet_path)['model'], strict=True)
        img1_tpb_net.load_state_dict(torch.load(img1_TPBNet_path, weights_only=True)['model'], strict=True)
        img2_tpb_net.load_state_dict(torch.load(img2_TPBNet_path, weights_only=True)['model'], strict=True)
        task_prompt_fusion_net.load_state_dict(torch.load(task_prompt_fusion_net_path, weights_only=True)['model'], strict=True)
        text2image_prompt_fusion_net.load_state_dict(torch.load(text2image_prompt_fusion_net_path, weights_only=True)['model'], strict=True)
        feature_fusion_net.load_state_dict(torch.load(feature_fusion_net_path, weights_only=True)['model'], strict=True)
 
    scb_net.eval()
    img1_tpb_net.eval()
    img2_tpb_net.eval()
    task_prompt_fusion_net.eval()
    text2image_prompt_fusion_net.eval()
    feature_fusion_net.eval()


    # Step-3: prepare data
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder=os.path.join(args.img_path,'img1')
    img2_image_folder=os.path.join(args.img_path,'img2')

    img1_image_path =os.path.join(img1_image_folder, args.file_name)
    img2_image_path =os.path.join(img2_image_folder, args.file_name)

    text_prompt=args.text_prompt

    img1=load_image(img1_image_path)
    img2=load_image(img2_image_path)

    # pil222_image = image222.copy()
    

    with torch.no_grad():
        # TPB
        img1_clip_visual_input = clip_image_processor(images=img1, return_tensors="pt").pixel_values.to(device=vae.device)
        img2_clip_visual_input = clip_image_processor(images=img2, return_tensors="pt").pixel_values.to(device=vae.device)
        img1_prompt_embeds = img1_tpb_net(clip_vision_outputs=clip_v(img1_clip_visual_input, output_attentions=True, output_hidden_states=True),
                                use_global=args.used_clip_vision_global,
                                layer_ids=args.used_clip_vision_layers,)
        img2_prompt_embeds = img2_tpb_net(clip_vision_outputs=clip_v(img2_clip_visual_input, output_attentions=True, output_hidden_states=True),
                                          use_global=args.used_clip_vision_global,
                                          layer_ids=args.used_clip_vision_layers,)
        fusion_prompt_guidance = task_prompt_fusion_net(img1_prompt_embeds, img2_prompt_embeds)

        # tmp1 = img1_prompt_embeds.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
        # tmp2 = img2_prompt_embeds.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
        # tmp3 = fusion_prompt_guidance.reshape(1, 16, 16, -1).permute(0, 3, 1, 2)
        # show_feature_map(tmp1, 1, show_plot=True)
        # show_feature_map(tmp2, 1, show_plot=True)
        # show_feature_map(tmp3, 1, show_plot=True)

        text = tokenizer(text_prompt, return_tensors="pt", padding="max_length", truncation=True).to(device=vae.device)

        text_features = text_encoder(**text, output_attentions=True, output_hidden_states=True).last_hidden_state
        fusion_prompt_guidance = text2image_prompt_fusion_net(fusion_prompt_guidance,text_features)

        # resolution adjustment (one can adjust this resolution also, as long as the short side is equal to or larger than 512)
        width, height = img1.size
        # if width < 512 or height < 512:
        #     if width < height:
        #         new_width = 512
        #         new_height = int((512 / width) * height)
        #     else:
        #         new_height = 512
        #         new_width = int((512 / height) * width)
        #     img1 = img1.resize((new_width, new_height))
        #     img2 = img2.resize((new_width, new_height))
        # else:
        #     new_height = height
        #     new_width = width


        # pre-process image
        img1 = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)  # image now is tensor in [-1,1]
        visualization_image_tensor(img1)
        img2 = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)  # image now is tensor in [-1,1]
        a= torch.chunk(vae.quant_conv(vae.encoder(img1)), 2, dim=1)[0]
        af = vae.decode(a, return_dict=False)[0]
        visualization_image_tensor(af)

        img1_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img1)), 2, dim=1)[0]
        img2_scb_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img2)), 2, dim=1)[0]



        # show_feature_map(img1_scb_cond,show_plot=True)
        # show_feature_map(img2_scb_cond,show_plot=True)

        fuse_scb_cond = feature_fusion_net(img1_scb_cond, img2_scb_cond)
        af = vae.decode(img1_scb_cond / vae.config.scaling_factor, return_dict=False)[0]
        af = vae_image_processor.postprocess(af, output_type='pil')[0]
        af.save(os.path.join("./Test_Results/init_fuse/", args.file_name))


        # show_feature_map(fuse_scb_cond,show_plot=True)
        b, c, h, w = fuse_scb_cond.size()

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
                latents = noise_scheduler.add_noise(fuse_scb_cond, noise, t, )

            # SCB
            down_block_res_samples = scb_net(
                latents,
                t,
                encoder_hidden_states=fusion_prompt_guidance,
                cond_img=fuse_scb_cond,
                return_dict=False,
            )


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
        save_.save(os.path.join(args.save_root,args.file_name))
    print('--------all done-----------')
