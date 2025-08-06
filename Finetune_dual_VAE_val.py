import os
import torch
import numpy as np
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from utils import get_latest_checkpoint, load_states, concat_imgs


def log_validation(logger, vae,encoder,decoder,feature_fusion_module,args, accelerator, step):
    logger.info("Running validation... ")

    image_logs = validation(args, vae,encoder,decoder,feature_fusion_module, args.validation_image, logger=logger)

    for tracker in accelerator.trackers:
        pred_list_for_save = []
        for log in image_logs:
            rec = log["rec"]
            img1 = log["img1"]

            pred_list_for_save += [rec]
            formatted_images = []

            formatted_images.append(np.asarray(img1.resize((rec.size))))
            formatted_images.append(np.asarray(rec))
            formatted_images = np.stack(formatted_images)
            tracker.writer.add_images("null", formatted_images, step, dataformats="NHWC")

    pred_save_path = os.path.join(args.output_dir, "visuals/pred")
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)
    concat_imgs(pred_list_for_save, target_size=pred_list_for_save[0].size).save(os.path.join(pred_save_path, str(step) + '.png'),
                                                                                 target_size=pred_list_for_save[0].size, target_dim=1)

    return image_logs


def validation(args, vae,encoder,decoder,feature_fusion_module,validation_images, logger=None):
    image_logs = []
    path = get_latest_checkpoint(args.output_dir)
    states_path = os.path.join(args.output_dir, path, "random_states_0.pkl")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    img1_image_folder = os.path.join(validation_images, 'img1')
    img2_image_folder = os.path.join(validation_images, 'img2')
    label_image_folder = os.path.join(validation_images, 'PSFusion_Fusion_results')
    # 获取文件夹中的所有文件
    img1_image_path = [
        os.path.join(img1_image_folder, file)
        for file in os.listdir(img1_image_folder)
        if os.path.basename(file) in ['00209D.png','00375D.png','00196D.png','01396D.png','01453D.png']
    ]

    img2_image_path = [
        os.path.join(img2_image_folder, file)
        for file in os.listdir(img2_image_folder)
        if os.path.basename(file) in ['00209D.png', '00375D.png', '00196D.png', '01396D.png', '01453D.png']
    ]

    label_image_path = [
        os.path.join(label_image_folder, file)
        for file in os.listdir(label_image_folder)
        if os.path.basename(file) in ['00209D.png', '00375D.png', '00196D.png', '01396D.png', '01453D.png']
    ]

    img1_image_path.sort(key=lambda x: os.path.basename(x))
    img2_image_path.sort(key=lambda x: os.path.basename(x))
    label_image_path.sort(key=lambda x: os.path.basename(x))


    for index, (img1 ,img2,label) in enumerate(zip(img1_image_path,img2_image_path,label_image_path)):
        img1 = load_image(img1)
        img2 = load_image(img2)
        label = load_image(label)
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)
        # 进行验证（冻结参数状态下）
        # 假设你有一个验证函数 validate()，进行模型的评估
        vae.eval()
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            load_states(states_path)
            width, height = img1.size
            img1_preprocess = vae_image_processor.preprocess(img1, height=height, width=width).to(device=vae.device)
            img2_preprocess = vae_image_processor.preprocess(img2, height=height, width=width).to(device=vae.device)
            label_preprocess = vae_image_processor.preprocess(label, height=height, width=width).to(device=vae.device)
            z1, feature1_list = encoder(img1_preprocess, return_feature=True)
            z2, feature2_list = encoder(img2_preprocess, return_feature=True)
            z_label = encoder(label_preprocess, return_feature=False)
            z_label = vae.config.scaling_factor * torch.chunk(vae.quant_conv(z_label), 2, dim=1)[0]

            feature1_list = [feat / vae.config.scaling_factor for feat in feature1_list]
            feature2_list = [feat / vae.config.scaling_factor for feat in feature2_list]

            feature_list = feature_fusion_module(feature1_list, feature2_list)
            rec = decoder(vae.post_quant_conv(z_label / vae.config.scaling_factor), feature_list)
            rec = vae_image_processor.postprocess(rec, output_type='pil')[0]

            generator = torch.Generator()
            generator.manual_seed(args.seed)

            image_logs.append({"img1": load_image(img1), "rec": rec,})

    return image_logs
