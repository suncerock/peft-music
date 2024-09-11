import yaml

import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data import DataModule
from models import ALL_MODELS


def train(config):
    with open(config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"], workers=True)
    exp = config["exp"]
    model_name = config["model_name"]

    data_cfg = config["data"]
    model_cfg = config["model"]
    trainer_cfg = config["trainer"]

    datamodule = DataModule(exp=exp, **data_cfg)
    model = ALL_MODELS[model_name](exp=exp, **model_cfg)

    # n_params, n_trainable_params = 0, 0
    # for name, param in model.named_parameters():
    #     print(name, param.shape, param.requires_grad)
    #     if name.startswith("proj"):
    #         continue
    #     n_params += torch.prod(torch.tensor(param.shape))
    #     if param.requires_grad:
    #         n_trainable_params += torch.prod(torch.tensor(param.shape))
    # print("Number of parameters: ", n_params / 1e6)
    # print("Number of trainable parameters: ", n_trainable_params)

    trainer = pl.Trainer(
        **trainer_cfg["args"],
        logger=TensorBoardLogger(**trainer_cfg["logger"]),
        callbacks=[
            ModelCheckpoint(**trainer_cfg["checkpoint"]),
        ]
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")

    return

if __name__ == '__main__':
    import fire

    fire.Fire(train)