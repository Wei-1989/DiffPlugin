import os
import math
import logging
import argparse
import sys
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, save

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor

import transformers
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

from val_V2L_modulate import log_validation
from utils import save_random_states
from dataset.Fusiondataset_withText import make_VIFusion_train_dataset, collate_fn
from utils import get_latest_checkpoint, save_args, code_backup
from Myutils import visualization_tensor, manage_checkpoints

from modules import TPBNet, TaskPromptFusionNet, Text2ImagePromptFusionModule,SCBNet,TextAdapter,V2LModulate

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Diff-Plugin training script.")
    parser.add_argument("--output_dir", type=str, default="./results/VIFusion/v2lModulate_0407a", )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--validation_steps", type=int, default=500,)
    parser.add_argument("--train_data_file_list", type=str, default='data/train/DiS_category.csv',)
    parser.add_argument("--used_clip_vision_layers", type=int, default=24,)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--use_vae_refine", action="store_true", default=False, help="Batch size (per device) for the training dataloader.")

    parser.add_argument('--project_path', type=str, default="./")
    parser.add_argument('--data_root', type=str, default="./TrainData")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4", required=False, )
    parser.add_argument("--clip_path", type=str, default="openai/clip-vit-large-patch14", required=False, )

    parser.add_argument("--cache_dir", type=str, default="./cache", )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--inp_of_unet_is_random_noise", action="store_true", default=False,
                        help="only set this to True for lowlight and highlight tasks")
    parser.add_argument("--time_threshold", type=int, default=960, help='this is used when we set the initial noise as inp+noise')
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--revision", type=str, default=None, required=False, )

    parser.add_argument("--max_train_steps", type=int, default=None, )

    parser.add_argument("--checkpoints_total_limit", type=int, default=5, )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )

    parser.add_argument("--learning_rate", type=float, default=1e-5, )
    parser.add_argument("--scale_lr", action="store_true", default=True, )
    parser.add_argument("--lr_scheduler", type=str, default="constant", )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=4, )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--logging_dir", type=str, default="logs", )
    parser.add_argument("--allow_tf32", action="store_true", )
    parser.add_argument("--report_to", type=str, default="tensorboard", )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], )
    parser.add_argument("--set_grads_to_none", action="store_true", )

    parser.add_argument("--validation_image", type=str, default="./TestData", help="List of validation image paths")

    parser.add_argument("--num_inference_steps", type=int, default=20, help=("diffusion steps for inference process"), )
    parser.add_argument("--tracker_project_name", type=str, default="lowlight", help="the name of dataset/task, e.g., derain, desnow")
    parser.add_argument("--used_clip_vision_global", action="store_true", default=False, )

    parser.add_argument("--down_block_types", type=str, nargs="+", default=["My_CrossAttnDownBlock2D","My_CrossAttnDownBlock2D",
                                                                            "My_CrossAttnDownBlock2D","DownBlock2D"], )
    # parser.add_argument("--down_block_types", type=str, nargs="+", default="CrossAttnDownBlock2D" )
    parser.add_argument("--block_out_channels", type=int, nargs="+", default=[320,640,1280,1280])
    parser.add_argument("--load_weights_from_unet", action="store_true", default=True, help='when change plugin position, this will be false')

    parser.add_argument("--num_cross_proj_layers", type=int, default=1, help='the number of projection layers for cross-att')
    parser.add_argument("--clip_v_dim", type=int, default=1024, choices=[768, 1024], help='the dim of last layer of the pre-trained clip-v')
    parser.add_argument("--use_data_aug", action="store_true", default=False, help="use data augmentation or not")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    ckpt_dir = r"./results/VIFusion/Hadamard_0401a/pick-checkpoint-13500/"
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        save_args(args)
        code_backup(args)
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "visuals"), exist_ok=True)

    vae = AutoencoderKL.from_pretrained('./pretrained-large-modal/VAE', subfolder="vae", revision=None)
    vae.eval()

    # SCBNet
    backup_unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None)
    backup_unet.config.down_block_types = args.down_block_types
    backup_unet.config.block_out_channels = args.block_out_channels
    backup_unet.config.in_channels = 4

    scb_net = SCBNet.from_unet(backup_unet, load_weights_from_unet=False)
    scb_net.load_state_dict(torch.load(os.path.join(ckpt_dir, "scb_net.pt"), weights_only=True)['model'], strict=True)
    scb_net.eval()

    unet = UNet2DConditionModel.from_pretrained("./pretrained-large-modal/unet", subfolder="unet", revision=None)
    unet.eval()

    textAdapter = TextAdapter()
    textAdapter.load_state_dict(torch.load(os.path.join(ckpt_dir, "textAdapter.pt"), weights_only=True)['model'], strict=True)
    textAdapter.eval()


    image_encoder = CLIPVisionModel.from_pretrained("./pretrained-large-modal/clip_vision")
    text_encoder = CLIPTextModel.from_pretrained("./pretrained-large-modal/text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("./pretrained-large-modal/tokenizer")

    v2lModulate = V2LModulate(2, clip_v_dim=1024,clip_text_dim=768)
    v2lModulate.train()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32.")

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
        logger.info('----------The true learning rate is {} ----------'.format(args.learning_rate))

    # Optimizer creation
    optimizer_class = torch.optim.AdamW
    params_to_optimize = []
    params_to_optimize.append({'params': v2lModulate.parameters(), 'lr': args.learning_rate})
    assert len(params_to_optimize) > 0, "No trainable parameters found. Make sure to have at least one of the models enabled."
    optimizer = optimizer_class(params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    # dataset and dataloader
    train_dataset = make_VIFusion_train_dataset(args, accelerator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size,
                                                   num_workers=args.dataloader_num_workers, )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    """
    When the gradient_accumulation_steps option is used, the max_train_steps will be automatically calculated 
    according to the number of epochs and the length of the training dataset
    """
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # scheduler can be obtained from diffusers.optimization

    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, lr_scheduler, v2lModulate, = accelerator.prepare(optimizer,train_dataloader,lr_scheduler,v2lModulate,)

    # For mixed precision training we cast the vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16


    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    scb_net.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    textAdapter.to(accelerator.device, dtype=weight_dtype)
    v2lModulate.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_image")
        tracker_config.pop("down_block_types")
        tracker_config.pop("block_out_channels")
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            path = get_latest_checkpoint(args.output_dir)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if os.path.exists(args.resume_from_checkpoint):
                accelerator.load_state(args.resume_from_checkpoint)
                print('load_state successfully-----------from: ', args.resume_from_checkpoint)
            else:
                # copy the checkpoint to the output_dir and do necessary changes
                accelerator.load_state(os.path.join(args.output_dir, path))
                print('load_state successfully-----------from: ', os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])  # for example, checkpoint-1000

            initial_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")
    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")
    logger.info("***** -------------Note the code for accelerator.accumulate----------------- *****")

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # TPB
            image_mean = torch.tensor(OPENAI_CLIP_MEAN).view(1, 3, 1, 1).to(accelerator.device,
                                                                            dtype=weight_dtype)
            image_std = torch.tensor(OPENAI_CLIP_STD).view(1, 3, 1, 1).to(accelerator.device,
                                                                          dtype=weight_dtype)
            img1_normalized_pixel_values = (batch["img1_pixel_values"].to(dtype=weight_dtype) + 1.0) / 2.0
            img1_normalized_pixel_values = torch.nn.functional.interpolate(img1_normalized_pixel_values, size=(224, 224), mode="bilinear",
                                                                           align_corners=False)
            img1_normalized_pixel_values = (img1_normalized_pixel_values - image_mean) / image_std

            img2_normalized_pixel_values = (batch["img2_pixel_values"].to(dtype=weight_dtype) + 1.0) / 2.0
            img2_normalized_pixel_values = torch.nn.functional.interpolate(img2_normalized_pixel_values, size=(224, 224), mode="bilinear",
                                                                           align_corners=False)
            img2_normalized_pixel_values = (img2_normalized_pixel_values - image_mean) / image_std
            img1_visual_input = image_encoder(img1_normalized_pixel_values, output_attentions=True, output_hidden_states=True)
            img2_visual_input = image_encoder(img2_normalized_pixel_values, output_attentions=True, output_hidden_states=True)

            fusion_prompt_guidance = v2lModulate(img1_visual_input, img2_visual_input,topk=10)

            text = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
            text_features = text_encoder(**text, output_attentions=True, output_hidden_states=True).last_hidden_state

            t_f=text_features.mean(dim=1, keepdim=True)
            pfg=fusion_prompt_guidance.mean(dim=1, keepdim=True)


            mse_loss = F.mse_loss(pfg.float(), t_f.float(), reduction="mean")
            cos_loss = (1- (F.normalize(t_f, p=2, dim=-1) * F.normalize(pfg, p=2, dim=-1)).sum(dim=-1)).mean(dim=0)


            loss = mse_loss + cos_loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(v2lModulate.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0 or global_step == 1:
                        manage_checkpoints(base_dir=args.output_dir, enhancement=args.checkpointing_steps,
                                           max_checkpoints=args.checkpoints_total_limit)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        save_random_states(logger, save_path)
                        if global_step != 1:
                            save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
                            logger.info(f"Optimizer state saved in {os.path.join(save_path, 'optimizer.bin')}")
                            save(lr_scheduler.state_dict(), os.path.join(save_path, 'scheduler.bin'))
                            logger.info(f"Scheduler state saved in {os.path.join(save_path, 'scheduler.bin')}")

                            save({'model': v2lModulate.state_dict()}, os.path.join(save_path, 'v2lModulate.pt'))
                            logger.info(f"Saved v2lModulate to {save_path}")

                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        log_validation(logger=logger,
                                       vae=vae,
                                       scb_net=scb_net,
                                       unet=unet,
                                       textAdapter=textAdapter,
                                       clip_image_encoder=image_encoder,
                                       v2lModulate=v2lModulate,
                                       args=args, accelerator=accelerator,
                                       global_step=global_step,
                                       device=accelerator.device,)

                        v2lModulate.train()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()

    main(args)
