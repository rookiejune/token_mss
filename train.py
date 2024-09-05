import argparse
import yaml
from tokensep.utils import get_data_module, get_compression_model, get_transformer_model
from tokensep.pl_module.separation import LitTokenSeparation
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config_yaml, "r"), Loader=yaml.FullLoader)

    workspace = config['common_kwargs']['workspace']
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    data_module = get_data_module(config['data_module'])

    compress_model = get_compression_model(config['compression_model'])

    transformer_model = get_transformer_model(config['transformer_model'])

    pl_module = LitTokenSeparation(
        compression_model=compress_model,
        transformer_model=transformer_model,
        **config['pl_module']['kwargs'],
    )

    logger = TensorBoardLogger(save_dir=workspace/"lightning_logs", name=config['common_kwargs']['name'])
    checkpoint_callback = ModelCheckpoint(dirpath=workspace/"lightning_checkpoints", save_last=True)
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        **config['trainer']['kwargs'],
    )

    trainer.fit(model=pl_module, datamodule=data_module)
