import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        _summary_

        Args:
            x (_type_): (batch_size, num_tokens, dims)

        Returns:
            _type_: _description_
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenFormerEncoder(nn.Module):
    def __init__(
        self,
        num_quantizers,
        d_model,
        num_heads: int=1,
        dropout: float=0.1,
        dim_feedforward: int=2048,
        num_layers: int=2,
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_layers= num_layers

        self.quantizer_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.token_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)

        self.transformer_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
        for _ in range(self.num_layers)])

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor=None,
        is_causal: bool=False,
        ):
        """
        _summary_

        Args:
            src (_type_): (batch_size, num_quantizers, num_tokens, dims)

        Returns: (batch_size, num_tokens, dims)
        """
        # Attention on tokens.
        x0 = einops.rearrange(src, 'b q t d -> (b q) t d')
        x_token_attn, _ = self.token_attn(x0, x0, x0, need_weights=False)
        # (b q) t d
        # print(x_token_attn)

        # Attention on quantizers.
        x1 = einops.rearrange(x_token_attn, "(b q) t d -> (b t) q d", q=self.num_quantizers)
        _, quantizer_attn_weights = self.quantizer_attn(x1, x1, x1)
        # quantizer_attn_weights: (b t) q q
        quantizer_attn_weights: Tensor
        quantizer_attn_weights = F.normalize(quantizer_attn_weights.sum(dim=1), dim=-1)
        # quantizer_attn_weights: (b t) q

        # Weighted sum embeddings from different quantizers
        x2 = (x1 * quantizer_attn_weights[..., None]).sum(dim=1)
        x3 = einops.rearrange(x2, "(b t) d -> b t d", b=src.shape[0])
        # x3: b t d

        x3 = self.pe(x3)

        for layer in self.transformer_encoder_layers:
            x3 = layer(x3, src_mask=src_mask, is_causal=is_causal)
        return x3