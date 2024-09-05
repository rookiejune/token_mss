import yaml
import typing as T
import numpy as np


def float32_to_int16(x: np.float32) -> np.int16:

    x = np.clip(x, a_min=-1, a_max=1)

    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.int16) -> np.float32:

    return (x / 32767.0).astype(np.float32)


def get_data_module(config):
    if config['type'] == 'MetaDataModule':
        from tokensep.data.data_module import MetaDataModule, MetaDataset
        from tokensep.data.sampler import MetaSampler
        assert config['sampler']['type'] == 'MetaSampler'
        sampler = MetaSampler(**config['sampler']['kwargs'])
        assert config['dataset']['type'] == 'MetaDataset'
        dataset = MetaDataset(**config['dataset']['kwargs'])
        return MetaDataModule(
            sampler=sampler,
            dataset=dataset,
            **config['kwargs']
        )
    else:
        raise NotImplementedError


def get_compression_model(config):
    if config['type'] == "DAC":
        import dac
        model_path = dac.utils.download(model_type="44khz")
        model = dac.DAC.load(model_path)
        return model
    else:
        raise NotImplementedError


def get_transformer_model(config):
    if config['type'] == "TokenFormer":
        from tokensep.model.TokenFormer import TokenFormer
        return TokenFormer(**config['kwargs'])
    else:
        raise NotImplementedError
