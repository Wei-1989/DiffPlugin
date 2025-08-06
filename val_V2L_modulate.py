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
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision import transforms

from modules import VAE_ShallowFeatureFusionModule, CustomEncoder, CustomDecoder, SCBNet,TextAdapter


def log_validation(logger,vae=None,scb_net=None,unet=None,textAdapter=None, clip_image_encoder=None,v2lModulate=None,
                   args=None, accelerator=None, global_step=None,device=None):
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
        scb_net=scb_net,
        v2lModulate=v2lModulate,
        noise_scheduler=noise_scheduler,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        vae_shallow_feature_fusion_module=vae_shallow_feature_fusion_module,
        validation_images = args.validation_image,
        logger = logger,
        refine=args.use_vae_refine,
        device=device)

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


def validation(args, vae, unet, clip_image_encoder,scb_net, noise_scheduler, vae_encoder, vae_decoder,vae_shallow_feature_fusion_module,v2lModulate,
               validation_images, logger=None,refine=False,device=None):
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
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        v2lModulate.eval()

        with torch.no_grad():

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

            img1_visual_input = clip_image_encoder(img1_normalized_pixel_values, output_attentions=True, output_hidden_states=True)
            img2_visual_input = clip_image_encoder(img2_normalized_pixel_values, output_attentions=True, output_hidden_states=True)

            fusion_prompt_guidance = v2lModulate(img1_visual_input, img2_visual_input, topk=10)

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
