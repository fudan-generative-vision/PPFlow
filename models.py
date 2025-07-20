# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT/blob/main/models.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import bisect
from timm.models.vision_transformer import Mlp
from modules.attention import Attentionwithmask
from modules.patch_emb import PyramidPatchEmbed
from modules.patch_n_pack import pack_pos_emb, is_power_of_four, pack_shift_scale_gate, un_patch_n_pack
from modules.invertible_sample import ChannelInvertibleSampling


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def expand_tensor(c, token_length):
    return [x.unsqueeze(1).expand(-1, token_length, -1) for x in c]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core PPFlow Model                             #
#################################################################################

class PPFlowBlock(nn.Module):
    """
    A PPFlow block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attentionwithmask(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, token_length_list=None, x_duration=None, attention_mask=None, training=True):
        c = self.adaLN_modulation(c) # c shape [bz, 6*dimension]
        if training:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = pack_shift_scale_gate(c, x_duration, token_length_list, final=False)
        else:
            c = c.chunk(6, dim=1)
            [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] = expand_tensor(c, x.shape[1])

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attention_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PyramidFinalLayer(nn.Module):
    """
    The pyramid final layer of PPFlow.
    """
    def __init__(self, hidden_size, patch_size, out_channels, patch_level):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        if patch_level == 2:
            self.linear1 = nn.Linear(hidden_size, patch_size * patch_size * out_channels * 4, bias=True)

            self.pyramid_projs = {
                    '0': self.linear1,
                }
        elif patch_level == 3:
            self.linear1 = nn.Linear(hidden_size, patch_size * patch_size * out_channels * 4, bias=True)
            self.linear2 = nn.Linear(hidden_size, patch_size * patch_size * out_channels * 2, bias=True)

            self.pyramid_projs = {
                    '0': self.linear1,
                    '1': self.linear2,
                }

    def forward(self, x, c, token_length_list, x_duration=None, x_ratio=None, training=True):
        if training:
            bz = c.shape[0]
            shift, scale = pack_shift_scale_gate(self.adaLN_modulation(c), x_duration, token_length_list, final=True)
            x = modulate(self.norm_final(x), shift, scale)
            x = un_patch_n_pack(x, x_ratio, x_duration, token_length_list, bz)
            return [
                self.pyramid_projs.get(str(i), self.linear)(x_i)
                for i, x_i in enumerate(x)
            ]
        else:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
            linear_idx = token_length_list.index(x.shape[1])
            x = self.pyramid_projs.get(str(linear_idx), self.linear)(x)
            return x


class PPFlow(nn.Module):
    """
    Diffusion model with a Transformer backbone and pyramidal patchification.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        patch_level=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.patch_level = patch_level

        # ===Added parameters for packing training===
        self.target_token_length = (input_size // patch_size) ** 2
        self.target_channel = in_channels
        self.inv_sample = ChannelInvertibleSampling()
        self.t1 = 0.5
        if patch_level == 2:
            factors = [4, 1]
        elif patch_level == 3:
            factors = [4, 2, 1]
            self.t2 = 0.75
        self.token_length_list = [self.target_token_length // f for f in factors]
        self.channel_list = [self.target_channel * f for f in factors]
        # ============================================

        self.x_embedder = PyramidPatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, patch_level=patch_level)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.level_embedder = LabelEmbedder(patch_level, hidden_size, dropout_prob=0.0)
        num_patches = self.x_embedder.num_patches
        # Will use position-interpolated fixed sin-cos embedding:
        if patch_level == 2:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches // 4, hidden_size), requires_grad=False),
                                        nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)])
        elif patch_level == 3:
            self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches // 4, hidden_size), requires_grad=False),
                                            nn.Parameter(torch.zeros(1, num_patches // 2, hidden_size), requires_grad=False),
                                            nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)])

        self.blocks = nn.ModuleList([
            PPFlowBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = PyramidFinalLayer(hidden_size, patch_size, self.out_channels, patch_level=patch_level)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        for i in range(len(self.pos_embed)):
            if i == 0:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed[-1].shape[-1], int(self.x_embedder.num_patches ** 0.5), interpolate_w=2, interpolate_h=2)
            elif self.patch_level == 3 and i == 1:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed[-1].shape[-1], int(self.x_embedder.num_patches ** 0.5), interpolate_w=2, interpolate_h=1)
            else:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed[-1].shape[-1], int(self.x_embedder.num_patches ** 0.5), interpolate_w=1, interpolate_h=1)
            self.pos_embed[i].data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # Initialize pyramidpatch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj1.bias, 0)
        if self.patch_level == 3:
            w2 = self.x_embedder.proj2.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj2.bias, 0)


        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        # Initialize level embedding table:
        nn.init.normal_(self.level_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.linear1.weight, 0)
        nn.init.constant_(self.final_layer.linear1.bias, 0)
        if self.patch_level == 3:
            nn.init.constant_(self.final_layer.linear2.weight, 0)
            nn.init.constant_(self.final_layer.linear2.bias, 0)

    def unpatchify(self, x, c):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        p = self.x_embedder.patch_size[0]
        if is_power_of_four(c//2):
            h = w = int(x.shape[1] ** 0.5)
        else:
            w = int((x.shape[1] / 2) ** 0.5)
            h = 2 * w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y, x_duration, x_ratio, training):
        """
        Forward pass of PPFlow training.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        bz = len(x)
        x, attention_mask = self.x_embedder(x, self.token_length_list, x_duration, training=training)  # (N, T, D), where T = H * W / patch_size ** 2
        x +=  pack_pos_emb(self.pos_embed, bz, x_duration, self.token_length_list) 
        # ========Added level embeddding ======================
        boundaries = [round(d.item() * bz) for d in x_duration]
        counts = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries) - 1)]
        full_indices = torch.cat([
            torch.full((count,), fill_value=idx, device=x.device, dtype=torch.long)
            for idx, count in enumerate(counts)
        ])
        level_emb = self.level_embedder(full_indices, train=False)
        # =====================================================
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y + level_emb                    # (N, D)
        for block in self.blocks:
            x = block(x, c, self.token_length_list, x_duration, attention_mask)   # (N, T, D)
        x = self.final_layer(x, c, self.token_length_list, x_duration, x_ratio, training)   # (N, T, patch_size ** 2 * out_channels)
        x_res = [
            sample
            for batch, channels in zip(x, self.channel_list)
            for sample in (
                self.unpatchify(batch, channels * 2).chunk(2, dim=1)[0]
                if self.learn_sigma
                else self.unpatchify(batch, channels * 2)
            )
        ] # list (N, out_channels, H, W)
        del x
        torch.cuda.empty_cache()
        return x_res

    def forward_wo_cfg(self, x, t, y):
        """
        Forward pass of PPFlow inference.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if t[0] <= self.t1:
            x = self.inv_sample.down_square(x, 2)
        elif self.patch_level == 3 and t[0] <= self.t2:
            x = self.inv_sample.down_w(x, 2)
        
        x, attention_mask = self.x_embedder(x, self.token_length_list, channel_list=self.channel_list, training=False)
        x += self.pos_embed[self.token_length_list.index(x.shape[1])]  # (N, T, D), where T = H * W / patch_size ** 2
        t_emb = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, train=False)    # (N, D)
        level_emb = self.level_embedder(
            torch.full((x.shape[0],), self.token_length_list.index(x.shape[1]), 
            dtype=torch.long, device=x.device),
            train=False
        )
        c = t_emb + y + level_emb                    # (N, D)
        for block in self.blocks:
            x = block(x, c, attention_mask=attention_mask, training=False)             # (N, T, D)
        x = self.final_layer(x, c, self.token_length_list, training=False)             # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, x.shape[-1] // (self.patch_size**2))    # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        
        if t[0] <= self.t1:
            x = self.inv_sample.up_square(x, 2)
        elif self.patch_level == 3 and t[0] <= self.t2:
            x = self.inv_sample.up_w(x, 2)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scales):
        """
        Forward pass of PPFlow, but also batches the uncondiontal PPFlow forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if self.patch_level == 2:
            t_duration = [self.t1]
        elif self.patch_level == 3:
            t_duration = [self.t1, self.t2]
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_wo_cfg(combined, t, y)
        index = bisect.bisect_left(t_duration, t[0])
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scales[index] * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolate_w=1, interpolate_h=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(0, grid_size, interpolate_h, dtype=np.float32)
    grid_w = np.arange(0, grid_size, interpolate_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size // interpolate_w, grid_size // interpolate_h])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   PPFlow Configs                              #
#################################################################################

def PPFlow_XL_2(**kwargs):
    return PPFlow(depth=28, hidden_size=1152, patch_size=2, patch_level=2, num_heads=16, **kwargs)

def PPFlow_XL_3(**kwargs):
    return PPFlow(depth=28, hidden_size=1152, patch_size=2, patch_level=3, num_heads=16, **kwargs)

def PPFlow_L_2(**kwargs):
    return PPFlow(depth=24, hidden_size=1024, patch_size=2, patch_level=2, num_heads=16, **kwargs)

def PPFlow_L_3(**kwargs):
    return PPFlow(depth=24, hidden_size=1024, patch_size=2, patch_level=3, num_heads=16, **kwargs)

def PPFlow_B_2(**kwargs):
    return PPFlow(depth=12, hidden_size=768, patch_size=2, patch_level=2, num_heads=12, **kwargs)

def PPFlow_B_3(**kwargs):
    return PPFlow(depth=12, hidden_size=768, patch_size=2, patch_level=3, num_heads=12, **kwargs)

def PPFlow_S_2(**kwargs):
    return PPFlow(depth=12, hidden_size=384, patch_size=2, patch_level=2, num_heads=6, **kwargs)

def PPFlow_S_3(**kwargs):
    return PPFlow(depth=12, hidden_size=384, patch_size=2, patch_level=3, num_heads=6, **kwargs)

PPFlow_models = {
    'PPFlow_XL_2': PPFlow_XL_2,  'PPFlow_XL_3': PPFlow_XL_3, 
    'PPFlow_L_2':  PPFlow_L_2,   'PPFlow_L_3':  PPFlow_L_3,  
    'PPFlow_B_2':  PPFlow_B_2,   'PPFlow_B_3':  PPFlow_B_3, 
    'PPFlow_S_2':  PPFlow_S_2,   'PPFlow_S_3':  PPFlow_S_3, 
}
