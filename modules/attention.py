'''
This code is build on timm package.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


class Attentionwithmask(Attention):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :, :]  # [B, 1, N, N]
            attention_mask = attention_mask.expand(B, self.num_heads, N, N)  # [B, num_heads, N, N]

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=attention_mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if attention_mask is not None:
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x