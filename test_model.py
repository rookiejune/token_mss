import unittest
from tokensep.model.TokenFormer.encoder import TokenFormerEncoder
from tokensep.model.TokenFormer.model import TokenFormer
import torch
import torch.nn as nn


class TestTokenFormer(unittest.TestCase):
    @unittest.skipIf(True, "")
    def test1(self):
        f = TokenFormerEncoder(num_quantizers=2, d_model=4)
        x = torch.randn(1, 2, 3, 4)
        print(f(x).shape)

    @unittest.skipIf(True, "")
    def test2(self):
        f = nn.MultiheadAttention(4, num_heads=1, batch_first=True)
        x = torch.randn(1, 3, 4)
        mask = torch.tril(torch.ones(3, 3))
        y, attn = f(x, x, x)
        # print(y)
        print(attn)
        y, attn = f(x, x, x, attn_mask=mask)
        # print(y)
        print(attn)
        y, attn = f(x, x, x, attn_mask=nn.Transformer.generate_square_subsequent_mask(3),is_causal=True)
        # print(y)
        print(attn)
        y, attn = f(x, x, x, attn_mask=nn.Transformer.generate_square_subsequent_mask(3))
        # print(y)
        print(attn)
        print(nn.Transformer.generate_square_subsequent_mask(3))

    @unittest.skipIf(True, "")
    def test3(self):
        f = TokenFormer(
            num_quantizers=2,
            num_embeddings=10,
            d_model=4,
        )

        src = torch.randint(low=0, high=10, size=(1, 2, 4))
        tgt = torch.randint(low=0, high=10, size=(1, 2, 3))
        print(f(src, tgt).shape)

    def test4(self):
        f = TokenFormer(
            num_quantizers=1,
            num_embeddings=5,
            d_model=4,
        )
        x = torch.randint(low=0, high=4, size=(1, 1, 4))
        x[..., 0] = 2
        x[..., -1] = 3
        logits = f.infer(x, offset=2, refine=True)
        print(logits)
        print(logits.argmax(dim=-1))
        print(logits.shape)


if __name__ == "__main__":
    unittest.main()