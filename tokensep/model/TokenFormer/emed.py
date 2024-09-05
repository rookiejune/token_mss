import torch
from torch import Tensor
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_quantizers, *args, **kwargs):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                *args, **kwargs
            )
        for _ in range(num_quantizers)])

    def forward(self, sequence: Tensor) -> Tensor:
        """
        _summary_

        Args:
            sequence (_type_): (batch_size, num_quantizers, num_tokens)
        """
        assert sequence.shape[1] == self.num_quantizers
        return torch.stack(
            list(map(lambda i: self.embeddings[i](sequence[:, i]), range(self.num_quantizers))),
            dim=1
        )


if __name__ == '__main__':
    x = torch.randint(low=0, high=10, size=(1, 2, 2))
    print(x)
    f = Embedding(
        num_quantizers=2,
        num_embeddings=10,
        embedding_dim=4
    )
    print(f(x).shape)
    print(f.embeddings[0](x[:, 0, ...]))
