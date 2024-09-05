import torch
from torch import Tensor
import torch.nn as nn
from .emed import Embedding
from .encoder import TokenFormerEncoder
from .decoder import TokenFormerDecoder


class TokenFormer(nn.Module):
    def __init__(
        self,
        num_quantizers,
        num_embeddings,
        d_model,
        nhead: int=1,
        num_encoder_layers: int=2,
        num_decoder_layers: int=2,
        dim_feedforward: int=2048,
        dropout: float=0.1,
        ) -> None:
        super().__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings

        self.embedding = Embedding(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=d_model
        )

        self.encoder = TokenFormerEncoder(
            num_quantizers=num_quantizers,
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            num_layers=num_encoder_layers
        )

        self.decoder = TokenFormerDecoder(
            num_quantizers=num_quantizers,
            d_model=d_model,
            num_heads=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            num_layers=num_decoder_layers
        )

        self.classifers = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_embeddings),
        ) for _ in range(self.num_quantizers)])

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask=None,
        tgt_is_causal=False,
    ):
        """
        _summary_

        Args:
            src (Tensor): (batch_size, num_quantizers, num_src_tokens)

            tgt (Tensor): (batch_size, num_quantizers, num_tgt_tokens)
        """
        x_src = self.embedding(src)
        # (batch_size, num_quantizers, num_src_tokens, dims)
        # print("x_src: ", x_src.shape)

        memory = self.encoder(x_src)

        x_tgt = self.embedding(tgt)
        # print("x_tgt: ", x_tgt.shape)
        x = self.decoder(x_tgt, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        # (batch_size, num_tgt_tokens, dims)

        logits = torch.stack(
            list(map(lambda i: self.classifers[i](x), range(self.num_quantizers))),
            dim=1
        )
        return logits

    def infer(
        self,
        src: Tensor,
        offset: int,
        refine: bool=False,
        max_loops: int=4,
    ):
        """
        _summary_

        Args:
            src (Tensor): (batch_size, num_quantizers, num_src_tokens)

        Returns:
            tgt (Tensor): (batch_size, num_quantizers, num_tgt_tokens)
        """
        batch_size, num_quantizers, num_src_tokens = src.shape
        T = num_src_tokens - 2
        # Initialize `logits` with <beg>
        logits = torch.zeros((batch_size, num_quantizers, 1, self.num_embeddings))
        logits[..., offset] = torch.inf

        num_loops = 0
        while True:
            new_logits: Tensor
            new_logits = self(src, logits.argmax(-1))
            # (batch_size, num_quantizers, num_tgt_tokens, num_embeddings)
            new_logit = new_logits[..., -1, :]
            # (batch_size, num_quantizers, num_embeddings)
            # Choose the next token from (end, beg_0, ..., beg_k)
            new_pred = new_logit[..., offset+1:].argmax(dim=-1)
            # (batch_size, num_quanztiers)
            new_token = new_pred.flatten().bincount().argmax()
            # print(new_token)
            # Forcing `new_logit` to be a perfect logit.
            new_logit = torch.zeros((batch_size, num_quantizers, 1, self.num_embeddings))
            new_logit[..., new_token + offset + 1] = torch.inf
            # print(new_logit)
            # Update logits by concatenating `new_logit`
            logits = torch.cat([logits, new_logit], dim=-2)
            # (batch_size, num_quantizers, num_tgt_tokens + 1, num_embeddings)

            if new_token == 0 or num_loops == max_loops:
                break

            for i in range(T):
                new_logits = self(src, logits.argmax(-1))
                # (batch_size, num_quantizers, num_tgt_tokens, dims)
                new_logit = new_logits[..., -1, :]
                new_logit[..., offset:] = -torch.inf
                # Add `new_logit`
                logits = torch.cat([logits, new_logit[..., None, :]], dim=-2)

                # Update old logits
                if refine and i > 0:
                    p = 2 + num_loops * (T + 1)
                    # new_logits:
                    #
                    #   [<beg_k>, T tokens], <beg_k>, <x_0>, ..., <x_{i-1}>, <x_i>
                    #
                    #                                          p
                    # logits:                                  |
                    #   <beg>, [<beg_k>, T tokens], <beg_k>, <x_0>..., <x_{i-1}>
                    logits[..., p: p + i, :] = (
                        logits[..., p: p + i, :] + new_logits[..., p - 1: p - 1 + i, :]
                    ) / 2

            num_loops += 1

        return logits
