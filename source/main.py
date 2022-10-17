import data
import framework
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY, LightningCLI


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pipeline", choices=["full", "train", "test"])
        parser.add_argument("--exp_name")
        parser.add_argument("--version")
        parser.add_argument("--checkpoint")


MODEL_REGISTRY.register_classes(framework, pl.core.lightning.LightningModule)
DATAMODULE_REGISTRY.register_classes(data, pl.core.LightningDataModule)

cli = CustomLightningCLI(
    auto_registry=True,
    subclass_mode_model=True,
    subclass_mode_data=True,
    save_config_overwrite=True,
    run=False,
    trainer_defaults={
        "callbacks": ModelCheckpoint(
            filename="{epoch:02d}-{val_metric:.2f}",
            every_n_epochs=10,
            save_last=True,
        )
    },
)

cli.trainer.logger = pl.loggers.TensorBoardLogger(
    save_dir=cli.trainer.default_root_dir,
    name=cli.config["exp_name"],
    version=cli.config["version"],
    default_hp_metric=False,
)

if cli.config["pipeline"] == "full":
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])
    cli.trainer.test(
        cli.model,
        cli.datamodule,
        # ckpt_path='best'
    )
elif cli.config["pipeline"] == "train":
    cli.trainer.fit(cli.model, cli.datamodule, ckpt_path=cli.config["checkpoint"])
elif cli.config["pipeline"] == "test":
    cli.trainer.test(cli.model, cli.datamodule, cli.config["checkpoint"])
