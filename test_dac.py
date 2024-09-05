import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
import dac
import unittest
import matplotlib.pyplot as plt
from tokensep.pl_module.utils import calculate_sdr


class UniTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)

    @unittest.skipIf(True, "")
    def test_mp3(self):
        raw_path = "source/music/color_your_night.mp3"
        signal = AudioSignal(raw_path)
        print(signal)

        # x = self.model.preprocess(signal.audio_data, signal.sample_rate)

        code = self.model.compress(signal)

        code.save("source/token/color_your_night.dac")

        y = self.model.decompress(code)
        y.write("source/result/color_your_night.mp3")

        print(F.mse_loss(y.audio_data, signal.audio_data))

    @unittest.skipIf(True, "")
    def test_codes(self):
        x = dac.DACFile.load("codes/color_your_night.dac")

        print(x.codes.shape) # (channels, num_codebooks, tokens)
        # codes from 0 to 1024
        # [2, 9, 20384]
        # for i in range(9):
        plt.hist(x.codes[0, 0, :1000])

        plt.show()

    # @unittest.skipIf(True, "")
    def test_random_input(self):
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        model.eval()

        x = torch.randn(1, 1, 512 * 64)

        audio, sr = torchaudio.load("source/music/color_your_night.mp3")
        print(audio.shape)
        x = audio.mean(dim=0)[:512*64][None, None, :]

        z, codes, latents, _, _ = model.encode(x)

        print("z.shape: ", z.shape)
        print("codes shape: ", codes.shape)
        print("Max code: ", codes.max())

        y1 = model.decode(z)
        print("Error of decoding from z: ", F.mse_loss(y1, x))

        y2 = model.decode(model.quantizer.from_codes(codes)[0])
        print("Error of decoding from code: ", F.mse_loss(y2, x))


if __name__ == "__main__":
    unittest.main()
    # x = dac.DACFile.load("codes/color_your_night.dac")

    # print(x.codes.shape) # (channels, num_codebooks, tokens)

    # print(torch.histogram(x.codes[0, 0].float(), bins=100))
    # for i in range(9):
    # plt.hist(x.codes[0, 0, :1000], bins=100)

    # plt.show()