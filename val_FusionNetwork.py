import os
import torch
import numpy as np
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from utils import get_latest_checkpoint, load_states, concat_imgs


def log_validation(logger, vae, unet, image_encoder, scb_net, img1_tpb_net, img2_tpb_net, feature_fusion_net,
                   task_prompt_fusion_net, args, accelerator, step):
    logger.info("Running validation... ")

    scb_net = accelerator.unwrap_model(scb_net)
    img1_tpb_net = accelerator.unwrap_model(img1_tpb_net)
    img2_tpb_net = accelerator.unwrap_model(img2_tpb_net)
    feature_fusion_net = accelerator.unwrap_model(feature_fusion_net)
    task_prompt_fusion_net = accelerator.unwrap_model(task_prompt_fusion_net)
    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    image_logs = validation(args, vae, unet, image_encoder, scb_net, img1_tpb_net, img2_tpb_net, feature_fusion_net, task_prompt_fusion_net,
                            noise_scheduler, args.validation_image, logger=logger)

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
            tracker.writer.add_images("null", formatted_images, step, dataformats="NHWC")

    pred_save_path = os.path.join(args.output_dir, "visuals/pred")
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    concat_imgs(pred_list_for_save, target_size=pred_list_for_save[0].size).save(os.path.join(pred_save_path, str(step) + '.png'),
                                                                                 target_size=pred_list_for_save[0].size, target_dim=1)

    return image_logs


def validation(args, vae, unet, image_encoder, scb_net, img1_tpb_net, img2_tpb_net, feature_fusion_net, task_prompt_fusion_net, noise_scheduler,
               validation_images, logger=None):
    image_logs = []
    path = get_latest_checkpoint(args.output_dir)
    states_path = os.path.join(args.output_dir, path, "random_states_0.pkl")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(validation_images, 'img1')
    img2_image_folder = os.path.join(validation_images, 'img2')
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

    img1_image_path.sort(key=lambda x: os.path.basename(x))
    img2_image_path.sort(key=lambda x: os.path.basename(x))

    for index, (img1, img2) in enumerate(zip(img1_image_path, img2_image_path)):
        img1 = load_image(img1)
        img2 = load_image(img2)
        clip_image_processor = CLIPImageProcessor()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        scb_net.eval()
        img1_tpb_net.eval()
        img2_tpb_net.eval()
        feature_fusion_net.eval()
        task_prompt_fusion_net.eval()

        with torch.no_grad():
            img1_input = clip_image_processor(images=img1, return_tensors="pt").pixel_values.to(device=vae.device)
            img2_input = clip_image_processor(images=img2, return_tensors="pt").pixel_values.to(device=vae.device)
            img1_prompt_embeds = img1_tpb_net(clip_vision_outputs=image_encoder(img1_input, output_attentions=True, output_hidden_states=True),
                                              use_global=args.used_clip_vision_global,
                                              layer_ids=args.used_clip_vision_layers, )
            img2_prompt_embeds = img2_tpb_net(clip_vision_outputs=image_encoder(img2_input, output_attentions=True, output_hidden_states=True),
                                              use_global=args.used_clip_vision_global,
                                              layer_ids=args.used_clip_vision_layers, )

            fusion_prompt_guidance = task_prompt_fusion_net(img1_prompt_embeds, img2_prompt_embeds)

            load_states(states_path)

            width, height = img1.size
            if width < 512 or height < 512:
                if width < height:
                    new_width = 512
                    new_height = int((512 / width) * height)
                else:
                    new_height = 512
                    new_width = int((512 / height) * width)
                img1 = img1.resize((new_width, new_height))
                img2 = img2.resize((new_width, new_height))
            else:
                new_height = height
                new_width = width

            img1_preprocess = vae_image_processor.preprocess(img1, height=new_height, width=new_width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=new_height, width=new_width).to(device=vae.device)  # image now is
            # tensor in [-1,1]
            img1_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img1_preprocess)), 2, dim=1)[0]
            img2_cond = vae.config.scaling_factor * torch.chunk(vae.quant_conv(vae.encoder(img2_preprocess)), 2, dim=1)[0]

            fuse_scb_cond = feature_fusion_net(img1_cond, img2_cond)

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

                down_block_res_samples = scb_net(latents, t, encoder_hidden_states=fusion_prompt_guidance, cond_img=fuse_scb_cond,
                                                 return_dict=False, )
                noise_pred = unet(latents, t, encoder_hidden_states=fusion_prompt_guidance,
                                  down_block_additional_residuals=down_block_res_samples).sample

                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred_pil = vae_image_processor.postprocess(pred, output_type='pil')[0]

            image_logs.append({"img1": load_image(img1), "img2": load_image(img2), "pred": pred_pil, })

    return image_logs
