import torchaudio
import torch.utils.data as data
import unittest


class DataUnitTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # self.root = "/Users/zhuyin/Desktop/Dataset"
        self.root = "/home/zhuyin/dataset"

    @unittest.skipIf(True, "")
    def test_torch_musdb(self):
        print("Try to get a item from pytorch MUSDB_HQ...")
        dataset = torchaudio.datasets.MUSDB_HQ(
            root=self.root,
            subset="train",
            # download=True
        )
        # dataloader = data.DataLoader(dataset, batch_size=1)

        waveform, sr, num_frames, track_name = dataset[0]

        # print(len(dataset))

        # waveform, sr, num_frames, track_name = dataset[0]
        print(f"Get a waveform with a shape of {waveform.shape}, named {track_name}.")
        print("Done!")

    @unittest.skipIf(True, "")
    def test_pl_musdb(self):
        print("Try to get a item from pl_MUSDBHQ...")
        from tokensep.pl_module.data import MUSDBHQ
        dataset = MUSDBHQ(
            duration=1.,
            root=self.root,
            subset="test",
            # download=True
        )

        waveform = dataset[210]
        print(f"Get a waveform with a shape of {waveform.shape}...")

        front = dataset._load_sample(0)[0][..., 210*44100:]
        end = dataset._load_sample(1)[0][..., :211*44100 - 9265664]

        # print(front.shape)
        # print(end.shape)
        print("The item should be a combination,"
              f" the front difference is {(waveform[..., :front.shape[-1]] - front).sum()}"
              f" and the end difference is {(waveform[..., -end.shape[-1]:] - end).sum()}" )

        print("Done!")

    @unittest.skipIf(True, "")
    def test_dataloader(self):
        print("Try to get a batch from pl_MUSDBHQ...")

        from tokensep.pl_module.data import MUSDBHQ
        dataset = MUSDBHQ(
            chunk_length=512*128,
            # root="/Users/zhuyin/Desktop/Dataset",
            root=self.root,
            subset="test",
            stereo=False,
            # download=True
        )
        dataloader = data.DataLoader(
            dataset, batch_size=128
        )

        print(f"Batch is with a shape of {next(iter(dataloader)).shape}")
        for batch in dataloader:
            print(batch.shape)
        print("Done!")

    def test_data_module(self):
        from tokensep.data.sampler import MetaSampler
        sampler = MetaSampler(
            indexes_path="/home/zhuyin/workspace/indexes/musdbhq.pkl",
            source_types=["vocals", "bass", "drums", "other"],
            batch_size=2,
            step_per_epoch=100
        )

        from tokensep.data.data_module import MetaDataset, DataModule
        dataset = MetaDataset(
            input_channels=1
        )

        data_module = DataModule(
            sampler=sampler,
            dataset=dataset,
            num_workers=8,
            distributed=False
        )


if __name__ == '__main__':
    unittest.main()
