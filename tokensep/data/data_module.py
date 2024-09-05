from typing import Dict, List, NoReturn, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .augmentor import Augmentor
from .sampler import DistributedSamplerWrapper


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)


class MetaDataset:
    def __init__(
        self,
        input_channels: int,
        augmentations: Dict=None,
    ):
        r"""Used for getting data according to a meta.

        Args:
            input_source_types: list of str, e.g., ['vocals', 'accompaniment']
            target_source_types: list of str, e.g., ['vocals']
            input_channels: int
            augmentor: Augmentor
            segment_samples: int
        """
        self.input_channels = input_channels
        self.augmentor = None
        if augmentations is not None:
            self.augmentor = Augmentor(augmentations)

    def __getitem__(self, meta: Dict) -> Dict:
        r"""Return data according to a meta. E.g., an input meta looks like: {
            'vocals': [
                {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0},
                {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0},
            ],
            'bass': [
                {'hdf5_path': 'songA.h5', 'key_in_hdf5': 'bass', 'begin_sample': 0},
                {'hdf5_path': 'songB.h5', 'key_in_hdf5': 'bass', 'begin_sample': 0},
            ].
        }

        Then, for each source, audios from all metas will be remixed.

        Returns:
            data_dict: dict, e.g., {
                'vocals': (channels, segments_num),
                'bass': (channels, segements_num),
            }
        """
        data_dict = {}

        for source_type in meta.keys():
            # E.g., ['vocals', 'accompaniment']

            waveforms = []  # Audio segments to be mix-audio augmented.

            for m in meta[source_type]:
                # E.g., {
                #     'hdf5_path': '.../song_A.h5',
                #     'key_in_hdf5': 'vocals',
                #     'begin_sample': '13406400',
                #     'end_sample': 13538700,
                # }

                hdf5_path = m['hdf5_path']
                key_in_hdf5 = m['key_in_hdf5']
                bgn_sample = m['begin_sample']
                end_sample = m['end_sample']

                with h5py.File(hdf5_path, 'r') as hf:
                    waveform = int16_to_float32(
                        hf[key_in_hdf5][:, bgn_sample:end_sample]
                    )

                if self.augmentor:
                    waveform = self.augmentor(waveform, source_type)

                waveforms.append(waveform)
            # E.g., waveforms: [(input_channels, audio_samples), (input_channels, audio_samples)]

            # mix-audio augmentation
            data_dict[source_type] = self.match_waveform_to_input_channels(
                np.mean(waveforms, axis=0),
                input_channels=self.input_channels
            )
            # data_dict[source_type]: (input_channels, audio_samples)

        # data_dict looks like: {
        #     'voclas': (input_channels, audio_samples),
        #     'accompaniment': (input_channels, audio_samples)
        # }
        return data_dict

    def match_waveform_to_input_channels(
        self,
        waveform: np.array,
        input_channels: int,
    ) -> np.array:
        r"""Match waveform to channels num.

        Args:
            waveform: (input_channels, segments_num)
            input_channels: int

        Outputs:
            output: (new_input_channels, segments_num)
        """
        waveform_channels = waveform.shape[0]

        if waveform_channels == input_channels:
            return waveform

        elif waveform_channels < input_channels:
            assert waveform_channels == 1
            return np.tile(waveform, (input_channels, 1))

        else:
            assert input_channels == 1
            return np.mean(waveform, axis=0)[None, :]


def collate_fn(list_data_dict: List[Dict]) -> Dict:
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            {'vocals': (input_channels, segment_samples),
             'accompaniment': (input_channels, segment_samples),
             'mixture': (input_channels, segment_samples)
            },
            ...]

    Returns:
        data_dict: e.g. {
            'vocals': (batch_size, input_channels, segment_samples),
            'accompaniment': (batch_size, input_channels, segment_samples),
            'mixture': (batch_size, input_channels, segment_samples)
            }
    """
    data_dict = {}
    # print(list_data_dict)
    for key in list_data_dict[0].keys():
        data_dict[key] = torch.Tensor(
            np.array([data_dict[key] for data_dict in list_data_dict])
        )
    return data_dict


class MetaDataModule(LightningDataModule):
    def __init__(
        self,
        sampler,
        dataset,
        num_workers: int,
        distributed: bool,
    ):
        r"""Data module.

        Args:
            train_sampler: Sampler object
            train_dataset: Dataset object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._sampler = sampler
        self.dataset = dataset
        self.num_workers = num_workers
        self.distributed = distributed

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # SegmentSampler is used for sampling segment indexes for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.

        if self.distributed:
            self.sampler = DistributedSamplerWrapper(self._sampler)

        else:
            self.sampler = self._sampler

    def train_dataloader(self) -> DataLoader:
        r"""Get train loader."""
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader
