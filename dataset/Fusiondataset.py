import os
import random
import torch
from torchvision import transforms

from datasets import load_dataset
from datasets import Image as DatasetsImage

from PIL import Image
import torchvision.transforms.functional as F
import random
import numpy as np
import cv2, math
from dataset.degradation import (random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression)

def batch_data_augmentation(input_batch, gt_batch):
    augmented_input_batch = []
    augmented_gt_batch = []
    
    for input_img, gt_img in zip(input_batch, gt_batch):
        aug_input, aug_gt = data_augmentation(input_img, gt_img)
        augmented_input_batch.append(aug_input)
        augmented_gt_batch.append(aug_gt)

    return augmented_input_batch, augmented_gt_batch


def data_augmentation(input_img, gt_img):
    input_img = Image.fromarray(input_img) if not isinstance(input_img, Image.Image) else input_img
    gt_img = Image.fromarray(gt_img) if not isinstance(gt_img, Image.Image) else gt_img

    w, h = input_img.size

    # Random Crop
    if random.random() > 0.5:
        if w > 512 and h > 512:
            top = random.randint(0, h - 512)
            left = random.randint(0, w - 512)
            input_img = F.crop(input_img, top, left, 512, 512)
            gt_img = F.crop(gt_img, top, left, 512, 512)
        elif w < 512 or h < 512:
            shorter_side = min(w, h)
            longer_side = max(w, h)
            start_pos = random.randint(0, longer_side - shorter_side)
            if w > h:
                input_img = F.crop(input_img, 0, start_pos, shorter_side, shorter_side)
                gt_img = F.crop(gt_img, 0, start_pos, shorter_side, shorter_side)
            else:
                input_img = F.crop(input_img, start_pos, 0, shorter_side, shorter_side)
                gt_img = F.crop(gt_img, start_pos, 0, shorter_side, shorter_side)

    # Random Horizontal Flip
    if random.random() > 0.5:
        input_img = F.hflip(input_img)
        gt_img = F.hflip(gt_img)

    # Resize to 512x512
    input_img = F.resize(input_img, (512, 512))
    gt_img = F.resize(gt_img, (512, 512))

    # Convert to Tensor and Normalize
    input_img = transforms.ToTensor()(input_img)
    gt_img = transforms.ToTensor()(gt_img)
    input_img = transforms.Normalize([0.5], [0.5])(input_img)
    gt_img = transforms.Normalize([0.5], [0.5])(gt_img)

    return input_img, gt_img

def apply_degradation(hqs, 
                      blur_kernel_size=41, 
                      kernel_list=['iso', 'aniso'], 
                      kernel_prob=[0.5, 0.5], 
                      blur_sigma=[0.1, 10], 
                      downsample_range=[0.8, 8], 
                      noise_range=[0, 20], 
                      jpeg_range=[60, 100]):
    """
    Input a list of PIL images, output a list of PIL images
    """

    lqs = []
    for pil_img_gt in hqs:
        img_gt = np.array(pil_img_gt)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        h, w, _ = img_gt.shape
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            blur_kernel_size,
            blur_sigma,
            blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(downsample_range[0], downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, noise_range)
        # jpeg compression
        if jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # convert to PIL image and append to the list
        img_lq = (img_lq * 255.0).round().clip(0, 255).astype(np.uint8)
        img_lq = Image.fromarray(img_lq[..., ::-1])
        lqs.append(img_lq)
    return lqs


def make_VIFusion_train_dataset(args, accelerator):
    dataset = load_dataset('csv',
                           data_files=args.train_data_file_list,
                           cache_dir=args.cache_dir, )

    dataset['train'] = dataset['train'].map(lambda example: {'label': os.path.join(args.data_root, example['label'])})
    dataset['train'] = dataset['train'].map(lambda example: {'img1': os.path.join(args.data_root, example['img1'])})
    dataset['train'] = dataset['train'].map(lambda example: {'img2': os.path.join(args.data_root, example['img2'])})

    dataset = dataset.cast_column("label", DatasetsImage())  # corresponding to the ground truth
    dataset = dataset.cast_column("img1", DatasetsImage())  # corresponding to the vi image
    dataset = dataset.cast_column("img2", DatasetsImage())  # corresponding to the ir image

    column_names = dataset["train"].column_names
    label_column = column_names[0]
    img1_column = column_names[1]
    img2_column = column_names[2]

    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        label_list = [image.convert("RGB") for image in examples[label_column]]
        img1_list = [image.convert("RGB") for image in examples[img1_column]]
        img2_list = [image.convert("RGB") for image in examples[img2_column]]

        # if 'face' in args.tracker_project_name:
        #     # means that this kind of data augmentation is only for face restoration task
        #     conditioning_images = apply_degradation(conditioning_images)

        # if args.use_data_aug:
        #     conditioning_images, images = batch_data_augmentation(conditioning_images, images)
        # else:
        label_list = [image_transforms(image) for image in label_list]
        img1_list = [image_transforms(image) for image in img1_list]
        img2_list = [image_transforms(image) for image in img2_list]

        examples["label_list"] = label_list
        examples["img1_list"] = img1_list
        examples["img2_list"] = img2_list

        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset
def collate_fn(examples):
    label_values = torch.stack([example["label_list"] for example in examples])
    label_values = label_values.to(memory_format=torch.contiguous_format).float()

    img1_pixel_values = torch.stack([example["img1_list"] for example in examples])
    img1_pixel_values = img1_pixel_values.to(memory_format=torch.contiguous_format).float()

    img2_pixel_values = torch.stack([example["img2_list"] for example in examples])
    img2_pixel_values = img2_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "label_values": label_values,
        "img1_pixel_values": img1_pixel_values,
        "img2_pixel_values": img2_pixel_values,
    }