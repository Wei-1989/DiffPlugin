import torch

from .package import *
import torch.nn.functional as F
from modules.My_CrossAttnDownBlock2D import My_CrossAttnDownBlock2D_0401a


class CrossModalFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossModalFusion, self).__init__()
        self.textAdapter = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels)
        )
        self.factor_ir = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * 2)
        )
        self.fusion = nn.Conv2d(out_channels*2,out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self, vi, ir,text):
        text=self.textAdapter(text)
        b,c,h,w=vi.shape
        vi=vi.view(b,c,h*w).permute(0,2,1)
        ir=ir.view(b,c,h*w).permute(0,2,1)
        b,l,c=vi.shape
        map1_list=[]
        map2_list=[]
        score_map1=torch.matmul(vi, text.transpose(-1, -2)).permute(0,2,1).view(b,-1,h,w)
        score_map2=torch.matmul(ir, text.transpose(-1, -2)).permute(0,2,1).view(b,-1,h,w)
        for i in range(c):
            map1,map2 = torch.chunk(F.softmax(torch.concat([score_map1[:, i:i + 1, :, :], score_map2[:, i:i + 1, :, :]], dim=1), dim=1), chunks=2,
                                dim=1)
            map1_list.append(map1)
            map2_list.append(map2)

        map1_all = torch.cat(map1_list, dim=1)
        map2_all = torch.cat(map2_list, dim=1)

        map1_all=F.softmax(map1_all, dim=1)
        map2_all=F.softmax(map2_all, dim=1)


        vi = vi.permute(0, 2, 1).view(b, -1, h, w)
        ir = ir.permute(0, 2, 1).view(b, -1, h, w)

        visualization_tensor(map1_all)
        visualization_tensor(map2_all)

        vi = (map1_all.unsqueeze(2) * vi.unsqueeze(1)).sum(dim=1)
        ir = (map2_all.unsqueeze(2) * ir.unsqueeze(1)).sum(dim=1)

        # visualization_tensor(vi)
        # visualization_tensor(ir)

        out=vi+ir
        # visualization_tensor(out)
        return out

class SCBNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            projection_class_embeddings_input_dim: Optional[int] = None,
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
            global_pool_conditions: bool = False,
    ):
        super().__init__()

        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.init_vi_conv = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)
        self.init_ir_conv = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding)

        self.init_fusion=CrossModalFusion(768,320)


        # self.ir_text_transformer=Transformer2DModel()

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
        )

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type in ["DownBlock2D","CrossAttnDownBlock2D","ResnetDownsampleBlock2D",
                                       "AttnDownBlock2D","CrossAttnDownBlock2D","SimpleCrossAttnDownBlock2D","SkipDownBlock2D",
                                       "AttnSkipDownBlock2D","DownEncoderBlock2D","AttnDownEncoderBlock2D","KDownBlock2D","KCrossAttnDownBlock2D"] :
                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim[i],
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            elif down_block_type=="My_CrossAttnDownBlock2D":
                down_block= My_CrossAttnDownBlock2D_0401a(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    downsample_padding=downsample_padding,
                    cross_attention_dim=cross_attention_dim,
                    attn_num_head_channels=attention_head_dim[i],
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
            else:
                raise ValueError(f"{down_block_type} does not exist.")

            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

    @classmethod
    def from_unet(
            cls,
            unet: UNet2DConditionModel,
            controlnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
            load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate Controlnet class from UNet2DConditionModel.

        Parameters:
            unet (`UNet2DConditionModel`):
                UNet model which weights are copied to the ControlNet. Note that all configuration options are also
                copied where applicable.
        """
        controlnet = cls(
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:
            # controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            controlnet.init_vi_conv.load_state_dict(unet.conv_in.state_dict())
            controlnet.init_ir_conv.load_state_dict(unet.conv_in.state_dict())

            if controlnet.class_embedding:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())
            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)



            # controlnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)
            # since we want to remove some blocks, that will change the channels for mid_block accordingly, so here we comment out the above line

        return controlnet

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            cond_img: torch.FloatTensor,
            conditioning_scale: float = 1.0,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guess_mode: bool = False,
            return_dict: bool = True,
            step=0,
            type="random",
    ) -> Union[SCBNetOutput, Tuple]:
        basic = cond_img.chunk(2, dim=1)[0]
        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(basic.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = basic.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=basic.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(basic.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(basic.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=basic.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        vi_feature = self.init_vi_conv(cond_img.chunk(2, dim=1)[0])
        ir_feature = self.init_ir_conv(cond_img.chunk(2, dim=1)[1])
        b, c, h, w =vi_feature.shape

        sample=self.init_fusion(vi_feature,ir_feature,encoder_hidden_states)
        # torch.save(sample, f"ori_sample.pt")
        down_block_res_samples = (sample,)
        for i,downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    step=step,
                    CrossADB_idx=i,
                    type=type,
                )
                # torch.save(sample, f"pt/random/{step}_{i}sample.pt")

            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0

            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]

        if len(down_block_res_samples) != 12:
            # This is for the case when we remove the last two blocks, we should add zeros to the list so that the addition in UNet will work
            for i in range(12 - len(down_block_res_samples)):
                down_block_res_samples.append(torch.zeros(1, device=down_block_res_samples[-1].device))

        if not return_dict:
            return (down_block_res_samples)

        return SCBNetOutput(
            down_block_res_samples=down_block_res_samples
        )