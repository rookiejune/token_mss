import argparse
import os
import pickle

import h5py
import yaml
from pathlib import Path


def create_indexes(args):
    r"""Create and write out training indexes into disk. The indexes may contain
    information from multiple datasets. During training, training indexes will
    be shuffled and iterated for selecting segments to be mixed. E.g., the
    training indexes_dict looks like: {
        'vocals': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410}
            ...
        ]
        'accompaniment': [
            {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 0}
            {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 4410}
            ...
        ]
    }
    """

    # Arugments & parameters
    config_yaml = args.config_yaml

    # Read config file.
    with open(config_yaml, "r") as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)

    # Path to write out index.
    indexes_path = Path(config["indexes_path"])
    indexes_path.parent.mkdir(parents=True, exist_ok=True)

    source_types = config["source_types"].keys()
    # E.g., ['vocals', 'accompaniment']

    indexes_dict = {source_type: [] for source_type in source_types}
    # E.g., indexes_dict will looks like: {
    #     'vocals': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'vocals', 'begin_sample': 4410}
    #         ...
    #     ]
    #     'accompaniment': [
    #         {'hdf5_path': '.../songA.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 0}
    #         {'hdf5_path': '.../songB.h5', 'key_in_hdf5': 'accompaniment', 'begin_sample': 4410}
    #         ...
    #     ]
    # }

    # Get training indexes for each source type.
    for source_type in source_types:
        # E.g., ['vocals', 'bass', ...]

        print("--- {} ---".format(source_type))

        dataset_config = config["source_types"][source_type]
        # E.g., ['musdb18', ...]

        # Each source can come from mulitple datasets.
        for dataset_type in dataset_config.keys():
            print(f"Create indexes from {dataset_type}")
            key_in_hdf5 = dataset_config[dataset_type]["key_in_hdf5"]
            # E.g., 'vocals'

            # Traverse all packed hdf5 files of a dataset.
            for hdf5_path in Path(dataset_config[dataset_type]['hdf5s_dir']).iterdir():
                with h5py.File(hdf5_path, "r") as hf:
                    bgn_sample = 0
                    hop_samples = dataset_config[dataset_type]['hop_samples']
                    seg_samples = dataset_config[dataset_type]['seg_samples']
                    count = 0
                    while bgn_sample + seg_samples < hf[key_in_hdf5].shape[-1]:
                        indexes_dict[source_type].append({
                            'hdf5_path': hdf5_path,
                            'key_in_hdf5': key_in_hdf5,
                            'begin_sample': bgn_sample,
                        })
                        bgn_sample += hop_samples
                        count += 1

            print("{} indexes: {}".format(dataset_type, count))

        print(
            "Total indexes for {}: {}".format(
                source_type, len(indexes_dict[source_type])
            )
        )

    pickle.dump(indexes_dict, open(indexes_path, "wb"))
    print("Write index dict to {}".format(indexes_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_yaml", type=str, required=True, help="User defined config file."
    )

    # Parse arguments.
    args = parser.parse_args()

    # Create training indexes.
    create_indexes(args)
