import typing as T
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch._tensor import Tensor
import torch
import torchaudio
from torch.utils.data import DataLoader


class MUSDBHQ(torchaudio.datasets.MUSDB_HQ):
    def __init__(
        self,
        chunk_length: int,
        stereo: bool=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stereo = stereo
        self.chunk_length = chunk_length

        num_frames_list = [self._load_sample(n)[2] for n in range(len(self.names))]
        self.frame_indexes = torch.Tensor(num_frames_list).cumsum_(dim=0).long()

    def __getitem__(self, n: int) -> Tensor:
        start_frame = (n % len(self)) * self.chunk_length
        track_index = torch.searchsorted(self.frame_indexes, start_frame, side="right")

        actual_start_frame = start_frame
        if track_index > 0:
            actual_start_frame -= self.frame_indexes[track_index - 1]
        actual_end_frame = actual_start_frame + self.chunk_length

        if actual_end_frame <= self.frame_indexes[track_index]:
            item = super().__getitem__(track_index)[0][..., actual_start_frame: actual_end_frame]
        else:
            item = torch.cat([
                super().__getitem__(track_index)[0][..., actual_start_frame:],
                super().__getitem__(track_index + 1)[0][..., :actual_end_frame - self.frame_indexes[track_index]],
            ], dim=-1)

        if self.stereo == False:
            item = item.mean(dim=1, keepdim=True)
        return item

    def __len__(self) -> int:
        return self.frame_indexes[-1] // self.chunk_length


class LitMUSDBHQ(pl.LightningDataModule):
    def __init__(
        self,
        root: str="/Users/zhuyin/Desktop/Dataset",
        download: bool=False,
        chunk_length: int=512*128,
        stereo: bool=True,
        batch_size: int=1,
        num_workers: int=8,
        train_batch_size: int=None,
        val_batch_size: int=None,
        test_batch_size: int=None,
    ) -> None:
        super().__init__()
        self.root = root
        self.download = download
        self.chunk_length = chunk_length
        self.stereo = stereo

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size if train_batch_size else batch_size
        self.val_batch_size = val_batch_size if val_batch_size else batch_size
        self.test_batch_size = test_batch_size if test_batch_size else batch_size

    def prepare_data(self) -> None:
        if self.download:
            MUSDBHQ(
                chunk_length=self.chunk_length,
                stereo=self.stereo,
                root=self.root,
                download=True
            )

    def setup(self, stage:str):
        # sources: ["bass", "drums", "other", "vocals"]
        if stage == "fit":
            self.train_dataset = MUSDBHQ(
                chunk_length=self.chunk_length,
                stereo=self.stereo,
                root=self.root,
                subset="train",
                split="train",
            )

        if stage == "fit" or stage == "validate":
            self.val_dataset = MUSDBHQ(
                chunk_length=self.chunk_length,
                stereo=self.stereo,
                root=self.root,
                subset="train",
                split="validation",
            )

        if stage == "test":
            self.test_dataset = MUSDBHQ(
                chunk_length=self.chunk_length,
                stereo=self.stereo,
                root=self.root,
                subset="test",
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers)
