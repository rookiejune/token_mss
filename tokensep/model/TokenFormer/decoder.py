from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import einops


class TokenFormerDecoder(nn.Module):
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

        self.transformer_decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
        for _ in range(self.num_layers)])

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor=None,
        tgt_is_causal: bool=False,):
        """
        _summary_

        Args:
            src (_type_): (batch_size, num_quantizers, num_tokens, dims)

        Returns: (batch_size, num_tokens, dims)
        """
        # Attention on tokens.
        x0 = einops.rearrange(tgt, 'b q t d -> (b q) t d')
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
        x3 = einops.rearrange(x2, "(b t) d -> b t d", b=tgt.shape[0])
        # x3: b t d

        for layer in self.transformer_decoder_layers:
            x3 = layer(x3, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        return x3
