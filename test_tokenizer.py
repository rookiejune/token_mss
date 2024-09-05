import unittest
from tokensep.pl_module.tokenizer import Tokenizer
import dac
import torch
import torch.nn.functional as F
import torchaudio


class TestTokenizer(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        waveform, _ = torchaudio.load("source/music/A Classic Education - NightOwl/vocals.wav")
        # waveform: (2, time_steps)
        self.vocals = waveform.mean(dim=0, keepdim=True)[None, ...]
        waveform, _ = torchaudio.load("source/music/A Classic Education - NightOwl/bass.wav")
        # waveform: (2, time_steps)
        self.bass = waveform.mean(dim=0, keepdim=True)[None, ...]

    @unittest.skipIf(True, "")
    def test_tokenize(self):
        tokenizer = Tokenizer(
            model = self.model,
            source_map={
                "bass": 0,
                "vocals": 1,
            }
        )

        mixture = torch.randn(1, 1, 44100)
        labels = {
            "bass": torch.randn(1, 1, 44100),
            "vocals": torch.randn(1, 1, 44100),
        }
        x, y = tokenizer(mixture, labels)

        print(x.shape)
        print(y.shape)
        print(y[..., 0])
        print(y[..., 1])
        print(y[..., 88])
        print(y[..., -1])

    # @unittest.skipIf(True, "")
    def test_detokenize(self):
        tokenizer = Tokenizer(
            model = self.model,
            source_map={
                "bass": 0,
                "vocals": 1,
            }
        )

        vocals = self.vocals[..., :512 * 64]
        bass = self.bass[..., :512 * 64]
        mixture = vocals + bass

        labels = {
            "bass": bass,
            "vocals": vocals,
        }

        print("code: ", self.model.encode(mixture)[1].shape)
        x, y = tokenizer(mixture, labels)

        x_inv = tokenizer.detokenize(x)
        print("mixture loss: ", F.mse_loss(x_inv, mixture))
        y_inv = tokenizer.detokenize(y)

        for key in labels.keys():
            print(f"{key} loss: ", F.mse_loss(y_inv[key], labels[key]))

if __name__ == "__main__":
    unittest.main()
