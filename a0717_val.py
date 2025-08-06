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
from Myutils import visualization_tensor

from modules import VAE_ShallowFeatureFusionModule, CustomEncoder, CustomDecoder


def log_validation(logger, vae, unet, text_encoder, tokenizer,scb_net,args, accelerator, global_step):
    logger.info("Running validation... ")

    scb_net.eval()
    unet.eval()

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

    ## Validation------------------
    image_logs = []
    path = get_latest_checkpoint(args.output_dir)
    states_path = os.path.join(args.output_dir, path, "random_states_0.pkl")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(args.validation_image, 'vi')
    img2_image_folder = os.path.join(args.validation_image, 'ir')
    text_folder = os.path.join(args.validation_image, 'gt_category_no_commas')
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
    target_filenames = {"00081D", "00123D", "01315N"}
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

        with torch.no_grad():
            text_token = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).to('cuda')
            text_features = text_encoder(text_token.input_ids)[0]

            uncondition_text_token = tokenizer(
                "",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).to('cuda')
            uncondition_text_features = text_encoder(uncondition_text_token.input_ids)[0]

            load_states(states_path)

            width, height = img1.size
            img1_preprocess = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)

            z1 = vae_encoder(img1_preprocess, return_feature=args.use_vae_refine)
            z2 = vae_encoder(img2_preprocess, return_feature=args.use_vae_refine)
            if args.use_vae_refine:
                z1,feature1_list=z1
                z2,feature2_list=z2

            img1_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z1), 2, dim=1)[0]
            img2_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z2), 2, dim=1)[0]

            b, c, h, w = img1_cond.size()
            generator = torch.Generator()
            generator.manual_seed(args.seed)
            # set the noise or latents

            latents = torch.randn((1, 4, h, w), generator=generator).cuda()


            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()

            for _, t in enumerate(timesteps):

                down_block_res_samples = scb_net(
                    timestep=t,
                    encoder_hidden_states=text_features,
                    cond_img=torch.concat([img1_cond, img2_cond], dim=1),
                    return_dict=False,
                )

                uncondition_down_block_res_samples = scb_net(
                    timestep=t,
                    encoder_hidden_states=uncondition_text_features,
                    cond_img=torch.concat([torch.zeros_like(img1_cond).to('cuda'), torch.zeros_like(img1_cond).to('cuda')], dim=1),
                    return_dict=False,
                )
                down_block_res_samples = [torch.cat([u, c], dim=0) for u, c in zip(uncondition_down_block_res_samples, down_block_res_samples)]

                latent_input = torch.cat([latents] * 2)
                encoder_input = torch.cat([uncondition_text_features, text_features])

                noise_pred = unet(
                    latent_input,
                    t,
                    encoder_hidden_states=encoder_input,
                    down_block_additional_residuals=down_block_res_samples,
                    show=False,
                ).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                # 应用 Classifier-Free Guidance
                noise_pred = noise_pred_uncond + 1.5 * (noise_pred_text - noise_pred_uncond)

                # 更新 latent
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            feature_list = vae_shallow_feature_fusion_module(feature1_list, feature2_list)
            feature_list = [feature / vae.config.scaling_factor for feature in feature_list]
            if args.use_vae_refine:
                rec = vae_decoder(vae.post_quant_conv(latents / vae.config.scaling_factor), feature_list)
            else:
                rec = vae_decoder(vae.post_quant_conv(latents / vae.config.scaling_factor))
            rec = vae_image_processor.postprocess(rec, output_type='pil')[0]

            image_logs.append({"img1": load_image(img1), "img2": load_image(img2), "pred": rec})




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


if __name__ == '__main__':
    pass
