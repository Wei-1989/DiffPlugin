from Myutils import visualization_tensor,visualization_token
from diffusers.models.transformer_2d import Transformer2DModel
from diffusers.models.attention import BasicTransformerBlock
import torch
from typing import Any, Dict
import torch.nn.functional as F

def CrossAttention_hook_fn(attn, args, kwargs, output):
    """ 获取 `attn2` 的注意力图 """
    hidden_states = args[0]
    b, l, c = hidden_states.shape
    if l == 4800:
        h = 60
        w = 80
    elif l == 1200:
        h = 30
        w = 40
    elif l == 300:
        h = 15
        w = 20
    else:
        h=int(l**0.5)
        w=int(l**0.5)
    visualization_tensor(hidden_states.permute(0, 2, 1).view(1, -1, h, w))
    encoder_hidden_states = kwargs["encoder_hidden_states"]
    attention_mask = kwargs["attention_mask"]

    residual = hidden_states

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = attn.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    visualization_tensor(query.permute(0, 2, 1).view(1, -1, h, w))

    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    attn_map = attention_probs.mean(dim=0, keepdim=False)
    for token in attn_map.permute(1, 0):
        visualization_tensor(token.view(h, w), batch_dim_exit=False)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor
    visualization_token(hidden_states)

    return hidden_states


def CrossAttnDownBlock2D_hook_fn(self, args, kwargs, output):
    hidden_states=kwargs['hidden_states']
    temb=kwargs['temb']
    encoder_hidden_states=kwargs['encoder_hidden_states']
    attention_mask=kwargs['attention_mask']
    cross_attention_kwargs=kwargs.get('cross_attention_kwargs', None)
    encoder_attention_mask= kwargs.get('encoder_attention_mask', None)

    output_states = ()
    for resnet, attn in zip(self.resnets, self.attentions):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
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
            attn.register_forward_hook(Transformer2D_hook_fn, with_kwargs=True)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)

        output_states = output_states + (hidden_states,)

    # for o in output_states:
    #     visualization_tensor(o)

    return hidden_states, output_states

def Transformer2D_hook_fn(self,args,kwargs,output):
    hidden_states=args[0]
    b,c,h,w=hidden_states.shape
    encoder_hidden_states=kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None
    timestep=kwargs['timestep'] if 'timestep' in kwargs else None
    class_labels=kwargs['class_labels'] if 'class_labels' in kwargs else None
    cross_attention_kwargs=kwargs['cross_attention_kwargs'] if 'cross_attention_kwargs' in kwargs else None
    attention_mask=kwargs['attention_mask'] if 'attention_mask' in kwargs else None
    encoder_attention_mask=kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None
    return_dict=kwargs['return_dict'] if 'return_dict' in kwargs else None

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
    for block in self.transformer_blocks:
        block.register_forward_hook(My_BasicTransformer_hook_fn,with_kwargs=True)
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
        )

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


    return (output,)

def My_BasicTransformer_hook_fn(self,args,kwargs,output):
    hidden_states=args[0]
    attention_mask=kwargs.get('attention_mask',None)
    encoder_hidden_states=kwargs.get('encoder_hidden_states',None)
    encoder_attention_mask=kwargs.get('encoder_attention_mask',None)
    timestep=kwargs.get('timestep',None)
    cross_attention_kwargs=kwargs.get('cross_attention_kwargs',None)
    class_labels=kwargs.get('class_labels',None)

    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    # visualization_token(hidden_states,save_path="./hidden_states.png")
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    # visualization_token(attn_output,h_w_ratio=1)
    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states
    # visualization_token(hidden_states,save_path="./after_self.png")

    # 2. Cross-Attention
    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        # visualization_token(attn_output,h_w_ratio=1)
        hidden_states = attn_output + hidden_states
        # visualization_token(hidden_states,save_path="./after_cross.png")

    # 3. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states
def My_Transformer2D_hook_fn(self,args,kwargs,output):
    hidden_states=args[0]
    b,c,h,w=hidden_states.shape
    encoder_hidden_states=kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None
    timestep=kwargs['timestep'] if 'timestep' in kwargs else None
    class_labels=kwargs['class_labels'] if 'class_labels' in kwargs else None
    cross_attention_kwargs=kwargs['cross_attention_kwargs'] if 'cross_attention_kwargs' in kwargs else None
    attention_mask=kwargs['attention_mask'] if 'attention_mask' in kwargs else None
    encoder_attention_mask=kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None
    return_dict=kwargs['return_dict'] if 'return_dict' in kwargs else None

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
    for block in self.transformer_blocks:
        hidden_states = block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=class_labels,
        )

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

    return (output,)



