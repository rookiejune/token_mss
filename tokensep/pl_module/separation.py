from typing import Dict, List, Tuple
import typing as T

import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
# from ..model.TokenFormer import TokenFormer
import einops

from .utils import (
    calculate_sdr,
)
from .tokenizer import Tokenizer


class LitTokenSeparation(pl.LightningModule):
    def __init__(
        self,
        compression_model: nn.Module,
        transformer_model: nn.Module,
        source_map: T.Dict[str, int],
        source_weight: T.Dict[str, float],
        # loss_fn: Callable=nn.functional.cross_entropy,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            model: nn.Module
            loss: function
            learning_rate: float
            lr_lambda: function
        """
        super().__init__()

        self.compression_model = compression_model
        # Freeze compression model.
        # for param in self.compression_model.parameters():
        #     param.requires_grad = False
        self.compression_model.eval()

        self.transformer_model = transformer_model

        self.tokenizer = Tokenizer(
            model=self.compression_model,
            source_map=source_map
        )
        # self.loss_fn = loss_fn
        self.source_weight = source_weight

        # self.first_data_pair = None

    def preprocess_data(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        _summary_

        Args:
            batch: E.g. {
                "vocals": (batch_size, channels, samples),
                "bass: {batch_size, channels, samples}, ...
            }
        """
        labels = {}

        mixture = 0

        for source in self.source_weight:
            if torch.rand(1,) < self.source_weight[source]:
                labels[source] = batch[source]
                mixture += batch[source]
        return mixture, labels

    def training_step(self, batch: Dict, _) -> Dict:
        mixture, labels = self.preprocess_data(batch)

        # if self.first_data_pair is None:
        #     mixture, labels = self.preprocess_data(batch)
        #     self.first_data_pair = (mixture, labels)
        # else:
        #     mixture, labels = self.first_data_pair

        src, tgt = self.tokenizer(mixture, labels)
        # src: batch_size, num_quantizers, num_src_tokens,
        # tgt: batch_size, num_quantizers, num_tgt_tokens

        logits = self.transformer_model(
            src,
            tgt[..., :-1],
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)),
            tgt_is_causal=True)
        # logits: batch_size, num_quantizers, num_tgt_tokens - 1, num_embeddings
        logits: Tensor

        loss = torch.nn.functional.cross_entropy(
            input=einops.rearrange(logits, "b q t e -> b e q t"),
            target=tgt[..., 1:],
        )

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: List[Dict], _) -> None:
        # mixture, labels = self.prepare_data(batch)

        # src, _ = self.tokenizer(mixture, labels)

        # self.transformer_model: TokenFormer
        # y = self.transformer_model.infer(src, offset=1024)

        # source_waveform_dict = self.tokenizer.detokenize(y)
        pass

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[Dict]]:
        optimizer = optim.Adam(
            self.transformer_model.parameters(),
            lr=1e-4
        )
        return optimizer
