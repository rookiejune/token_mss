import argparse
import time
import typing as T

import h5py
import librosa
import numpy as np
import torchaudio
from pathlib import Path
import yaml


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)

# Source types of the MUSDB18 dataset.
SOURCE_TYPES = ["vocals", "drums", "bass", "other", "accompaniment"]


def pack_audios_to_hdf5s(args):
    r"""Pack (resampled) audio files into hdf5 files to speed up loading.
    """

    # arguments & parameters
    config_yaml = args.config_yaml
    with open(config_yaml, "r") as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)
    hdf5s_dir = Path(config['hdf5s_dir'])
    hdf5s_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchaudio.datasets.MUSDB_HQ(
        root=config['root'],
        subset=config['subset'],
        download=False,
    )

    params = []  # A list of params for multiple processing.

    # Uncomment for debug.
    # write_single_audio_to_hdf5(params[0])
    # os._exit(0)

    pack_hdf5s_time = time.time()

    for data in dataset:
        waveform, sr, num_frames, name = data
        # waveform: (4, 2, )
        hdf5_path = hdf5s_dir/f"{name}.h5"
        with h5py.File(hdf5_path, "w") as hf:
            for i, source in enumerate(["bass", "drums", "other", "vocals"]):
                resampled_waveform = waveform[i]
                if (target_sr := config['sample_rate']) and sr != target_sr:
                    resampled_waveform = librosa.resample(
                        waveform[i],
                        orig_sr=sr,
                        target_sr=target_sr,
                        res_type=config['res_type'],
                    )

                hf.create_dataset(
                    name=source,
                    data=float32_to_int16(resampled_waveform.numpy()),
                    dtype=np.int16
                )
        print(f"Pack {name} to {hdf5_path} with a shape of {resampled_waveform.shape}")

    print("Pack hdf5 time: {:.3f} s".format(time.time() - pack_hdf5s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
    )

    # Parse arguments.
    args = parser.parse_args()

    # Pack audios into hdf5 files.
    pack_audios_to_hdf5s(args)
