from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import is_torch_version,deprecate
from diffusers.models.attention import AdaGroupNorm,BasicTransformerBlock,Attention
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.resnet import Downsample2D, FirDownsample2D, FirUpsample2D, KDownsample2D, KUpsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel,Transformer2DModelOutput
from Myutils.visualize import visualization_token

def Self_Attn(attn,hidden_states,attention_mask=None,temb=None,):
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = hidden_states.shape
    inner_dim = hidden_states.shape[-1]

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)


    encoder_hidden_states = hidden_states


    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # attn_map=query @ key.transpose(-2, -1)
    # w1=int(math.sqrt(int(attn_map.shape[2]/0.75)))
    # for head in attn_map[0]:
    #     for token in head.permute(1,0):
    #         visualization_tensor(token.view(int(0.75*w1),w1))



    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states

def Cross_Attn(attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,return_attn=False):
    # visualization_token(hidden_states,h_w_ratio=1,show_info=True)
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    inner_dim = hidden_states.shape[-1]

    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    # attn_map=F.softmax((query @ key.transpose(-2, -1)),dim=-1)
    attn_map=query @ key.transpose(-2, -1)
    # visualization_token(attn_map,type='BHTT',h_w_ratio=1)


    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    # visualization_token(hidden_states,h_w_ratio=1,show_info=True)
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    if return_attn:
        return hidden_states, attn_map
    return hidden_states

class My_BasicTransformerBlock(BasicTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        return_attn_map: bool = False,
        l_layer=None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # visualization_tensor(hidden_states.permute(0, 2, 1).view(1, -1, 60, 80))
        attn_output = Self_Attn(self.attn1,norm_hidden_states,attention_mask=attention_mask,)
        # visualization_tensor(attn_output.permute(0, 2, 1).view(1, -1, 60, 80))
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states
        # visualization_tensor(hidden_states.permute(0, 2, 1).view(1, -1, 60, 80))

        # visualization_tensor(hidden_states.permute(0, 2, 1).view(1, -1, 60, 80))

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            cross_attn_output=Cross_Attn(self.attn2,norm_hidden_states,encoder_hidden_states=encoder_hidden_states,
                                         attention_mask=encoder_attention_mask,return_attn=False)

            if l_layer is not None and l_layer == 0:
                cross_attn_output, attn_map = cross_attn_output
                visualization_token(attn_map,cmap='viridis')
            # visualization_tensor(cross_attn_output.permute(0, 2, 1).view(1, -1, 60, 80))
            hidden_states = hidden_states + cross_attn_output
            # visualization_tensor(hidden_states.permute(0, 2, 1).view(1, -1, 60, 80))


        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        # visualization_tensor(norm_hidden_states.permute(0, 2, 1).view(1, -1, 60, 80))
        return hidden_states


class My_Transformer2DModel(Transformer2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get('norm_type', 'layer_norm') == "layer_norm" and kwargs.get('num_embeds_ada_norm',None) is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"
        else:
            norm_type = kwargs.get('norm_type', 'layer_norm')


        self.transformer_blocks = torch.nn.ModuleList(
            [
                My_BasicTransformerBlock(
                    dim=kwargs.get('num_attention_heads', 320)*kwargs.get('attention_head_dim',40),
                    num_attention_heads=kwargs.get('num_attention_heads', 16),
                    attention_head_dim=kwargs.get('attention_head_dim', 40),
                    dropout=kwargs.get('dropout', 0.0),
                    cross_attention_dim=kwargs.get('cross_attention_dim', 768),
                    activation_fn=kwargs.get('activation_fn', 'geglu'),
                    num_embeds_ada_norm=kwargs.get('num_embeds_ada_norm', None),
                    attention_bias=kwargs.get('attention_bias', False),
                    only_cross_attention=kwargs.get('only_cross_attention', False),
                    upcast_attention=kwargs.get('upcast_attention', False),
                    norm_type=norm_type,
                    norm_elementwise_affine=kwargs.get('norm_elementwise_affine', True),
                )
                for d in range(kwargs.get('num_layers', 1))
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_attn_map: bool = False,
        l_layer=None,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            encoder_attention_mask ( `torch.Tensor`, *optional* ).
                Cross-attention mask, applied to encoder_hidden_states. Two formats supported:
                    Mask `(batch, sequence_length)` True = keep, False = discard. Bias `(batch, 1, sequence_length)` 0
                    = keep, -10000 = discard.
                If ndim == 2: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

        # 2. Blocks
        for i,block in enumerate(self.transformer_blocks):
            if l_layer is not None and l_layer==0:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    return_attn_map=return_attn_map,
                    l_layer=i,
                )
                if return_attn_map:
                    hidden_states,attn_map=hidden_states
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    return_attn_map=return_attn_map,
                )
                if return_attn_map:
                    hidden_states, attn_map = hidden_states

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            # TODO: cleanup!
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)

            # unpatchify
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )

        if not return_dict:
            if return_attn_map:
                return (output,attn_map)
            return (output,)

        return Transformer2DModelOutput(sample=output)

class MyCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    My_Transformer2DModel(
                        num_attention_heads=attn_num_head_channels,
                        attention_head_dim=out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    None,  # timestep
                    None,  # class_labels
                    cross_attention_kwargs,
                    attention_mask,
                    encoder_attention_mask,
                    **ckpt_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states