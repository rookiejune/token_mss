import unittest
import pytorch_lightning as pl
from tokensep.pl_module.separation import LitTokenSeparation
from tokensep.pl_module.data import LitMUSDBHQ
from tokensep.model.TokenFormer import TokenFormer

import dac


class TestLitTokenSepation(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        model_path = dac.utils.download(model_type="44khz")
        compression_model = dac.DAC.load(model_path)

        self.pl_module = LitTokenSeparation(
            compression_model=compression_model,
            transformer_model=TokenFormer(
                num_quantizers=9,
                num_embeddings=1030,
                d_model=256,
                nhead=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
            ),
            source_map={
                "bass": 0,
                "drums": 1,
                "other": 2,
                "vocals": 3
            },
            source_weight={
                'bass': 1/2,
                "drums": 1/2,
                "other": 1,
                "vocals": 3/4
            }
        )

        from tokensep.data.sampler import MetaSampler
        sampler = MetaSampler(
            indexes_path="/home/zhuyin/workspace/indexes/musdbhq.pkl",
            source_types=["vocals", "bass", "drums", "other"],
            batch_size=16,
            step_per_epoch=10000
        )

        from tokensep.data.data_module import MetaDataset, MetaDataModule
        dataset = MetaDataset(
            input_channels=1
        )

        self.data_module = MetaDataModule(
            sampler=sampler,
            dataset=dataset,
            num_workers=8,
            distributed=True
        )

    @unittest.skipIf(True, "")
    def test_remix_musdbhq(self):
        self.data_module.setup('fit')
        dataloader = self.data_module.train_dataloader()
        batch = next(iter(dataloader))
        mixture, labels = self.pl_module.preprocess_data(batch)
        print(mixture)
        print(labels)

    @unittest.skipIf(True, "")
    def test_data_module(self):
        # self.data_module.setup('fit')
        dataloader = self.data_module.train_dataloader()
        print(next(iter(dataloader)))

    # @unittest.skipIf(True, "")
    def test_training(self):
        trainer = pl.Trainer(
            # limit_predict_batches=100,
            max_epochs=1,
            strategy='ddp_find_unused_parameters_true',
            use_distributed_sampler=False
        )
        trainer.fit(model=self.pl_module, datamodule=self.data_module)

if __name__ == "__main__":
    unittest.main()
