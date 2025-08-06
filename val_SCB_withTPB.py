import os
import torch
import random
import numpy as np
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from utils import get_latest_checkpoint, load_states, concat_imgs
from Myutils.visualize_feature import show_feature_map
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
import json

from modules import VAE_ShallowFeatureFusionModule, CustomEncoder, CustomDecoder


def log_validation(logger, vae, unet, clip_image_encoder, text_encoder, tokenizer, text2image_prompt_fusion_net, img1_tpb_net,
                   img2_tpb_net, task_prompt_fusion_net,scb_net,
                   args, accelerator, global_step,textAdapter):
    logger.info("Running validation... ")

    scb_net.eval()
    unet.eval()
    textAdapter.eval()

    vae_pretrained_path = "./results/Stage_two/whole_network/pick-checkpoint-54000/"
    with open("./pretrained-large-modal/VAE/config.json", "r") as f:
        config = json.load(f)
    vae_encoder = CustomEncoder(config, vae.encoder).to(vae.device)
    vae_encoder.load_state_dict(torch.load(os.path.join(vae_pretrained_path, "encoder.pt"), weights_only=True)['model'], strict=True)
    vae_encoder.eval()
    vae_decoder = CustomDecoder(config, vae.decoder).to(vae.device)
    vae_decoder.load_state_dict(torch.load(os.path.join(vae_pretrained_path, "decoder.pt"), weights_only=True)['model'], strict=True)
    vae_decoder.eval()
    vae_shallow_feature_fusion_module = VAE_ShallowFeatureFusionModule([128, 128, 256, 512, 512]).to(vae.device)
    vae_shallow_feature_fusion_module.load_state_dict(torch.load(os.path.join(vae_pretrained_path, "feature_fusion_module.pt"), weights_only=True)[
                                                          'model'], strict=True)

    noise_scheduler = UniPCMultistepScheduler.from_pretrained("./pretrained-large-modal/scheduler", subfolder="scheduler")

    image_logs = validation(
        args=args,
        vae=vae,
        unet=unet,
        clip_image_encoder=clip_image_encoder,
        text_encoder = text_encoder,
        tokenizer=tokenizer,
        textAdapter=textAdapter,
        img1_tpb_net=img1_tpb_net,
        img2_tpb_net=img2_tpb_net,
        scb_net=scb_net,
        task_prompt_fusion_net=task_prompt_fusion_net,
        text2image_prompt_fusion_net=text2image_prompt_fusion_net,
        noise_scheduler=noise_scheduler,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        vae_shallow_feature_fusion_module=vae_shallow_feature_fusion_module,
        validation_images = args.validation_image,
        logger = logger)

    for tracker in accelerator.trackers:
        pred_list_for_save = []

        for log in image_logs:
            pred = log["pred"]
            img1 = log["img1"]
            img2 = log["img2"]

            pred_list_for_save += [pred]

            formatted_images = []

            formatted_images.append(np.asarray(img1.resize((pred.size))))
            formatted_images.append(np.asarray(img2.resize((pred.size))))
            formatted_images.append(np.asarray(pred))
            formatted_images = np.stack(formatted_images)
            tracker.writer.add_images("null", formatted_images, global_step, dataformats="NHWC")
    pred_save_path = os.path.join(args.output_dir, "visuals/pred")

    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    concat_imgs(pred_list_for_save, target_size=pred_list_for_save[0].size).save(os.path.join(pred_save_path, str(global_step) + '.png'),
                                                                                 target_size=pred_list_for_save[0].size, target_dim=1)

    return image_logs


def validation(args, vae, unet, clip_image_encoder, text_encoder, tokenizer,textAdapter, img1_tpb_net, img2_tpb_net,scb_net, task_prompt_fusion_net,
               text2image_prompt_fusion_net, noise_scheduler, vae_encoder, vae_decoder,vae_shallow_feature_fusion_module,
               validation_images, logger=None):
    image_logs = []
    path = get_latest_checkpoint(args.output_dir)
    states_path = os.path.join(args.output_dir, path, "random_states_0.pkl")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(validation_images, 'img1')
    img2_image_folder = os.path.join(validation_images, 'img2')
    text_folder = os.path.join(validation_images, 'category')
    # 获取文件夹中的所有文件
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
    target_filenames = {"00081D","00123D","01315N" }
    img1_image_path = [p for p in img1_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    img2_image_path = [p for p in img2_image_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    text_path = [p for p in text_path if os.path.splitext(os.path.basename(p))[0] in target_filenames]
    # -----------------------------

    for index, (img1, img2, text) in enumerate(zip(img1_image_path, img2_image_path, text_path)):
        img1 = load_image(img1)
        img2 = load_image(img2)
        with open(text, 'r', encoding='utf-8') as file:
            content = file.read()  # 读取整个文件内容
        clip_image_processor = CLIPImageProcessor()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        img1_tpb_net.eval()
        img2_tpb_net.eval()
        task_prompt_fusion_net.eval()

        with torch.no_grad():
            img1_input = clip_image_processor(images=img1, return_tensors="pt").pixel_values.to(device=vae.device)
            img2_input = clip_image_processor(images=img2, return_tensors="pt").pixel_values.to(device=vae.device)
            img1_prompt_embeds = img1_tpb_net(clip_vision_outputs=clip_image_encoder(img1_input, output_attentions=True, output_hidden_states=True),
                                              use_global=args.used_clip_vision_global,
                                              layer_ids=args.used_clip_vision_layers, )
            img2_prompt_embeds = img2_tpb_net(clip_vision_outputs=clip_image_encoder(img2_input, output_attentions=True, output_hidden_states=True),
                                              use_global=args.used_clip_vision_global,
                                              layer_ids=args.used_clip_vision_layers, )

            fusion_prompt_guidance = task_prompt_fusion_net(img1_prompt_embeds, img2_prompt_embeds)

            text = tokenizer(content, return_tensors="pt", padding=False, truncation=True).to(device=vae.device)
            text_features = text_encoder(**text, output_attentions=True, output_hidden_states=True).last_hidden_state
            text_features=textAdapter(text_features)
            # text_features = text_features[:, 1:-1, :]
            fusion_prompt_guidance, _ = text2image_prompt_fusion_net(fusion_prompt_guidance, text_features)

            load_states(states_path)

            width, height = img1.size
            img1_preprocess = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)

            z1, feature1_list = vae_encoder(img1_preprocess, return_feature=True)
            z2, feature2_list = vae_encoder(img2_preprocess, return_feature=True)

            img1_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z1), 2, dim=1)[0]
            img2_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z2), 2, dim=1)[0]

            b, c, h, w = img1_cond.size()
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            # set the noise or latents
            if args.inp_of_unet_is_random_noise:
                latents = torch.randn((1, 4, h, w), generator=generator).cuda()
            else:
                noise = torch.randn((1, 4, h, w), generator=generator).cuda()

            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()

            for _, t in enumerate(timesteps):

                if t >= args.time_threshold and not args.inp_of_unet_is_random_noise:
                    latents = noise_scheduler.add_noise(img1_cond, noise, t, )

                down_block_res_samples = scb_net(
                    timestep=t,
                    encoder_hidden_states=fusion_prompt_guidance,
                    cond_img=torch.concat([img1_cond, img2_cond], dim=1),
                    return_dict=False,
                    step=999999
                )

                noise_pred = unet(
                    latents,
                    t,
                    encoder_hidden_states=fusion_prompt_guidance,
                    down_block_additional_residuals=down_block_res_samples
                ).sample

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # feature_list = vae_shallow_feature_fusion_module(feature1_list, feature2_list)
            # feature_list = [feature / vae.config.scaling_factor for feature in feature_list]
            pred = vae_decoder(vae.post_quant_conv(latents / vae.config.scaling_factor))

            pred_pil = vae_image_processor.postprocess(pred, output_type='pil')[0]

            image_logs.append({"img1": load_image(img1), "img2": load_image(img2), "pred": pred_pil})

    return image_logs


if __name__ == '__main__':
    pass
