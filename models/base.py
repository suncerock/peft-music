import math
from typing import Any, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .loss_fn import ALL_LOSSES
from .metrics import build_metrics


N_OUTPUTS: Dict[str, int] = dict(
    mtat=50,

    mtg_top50=50,
    mtg_genre=95,
    mtg_instrument=41,
    mtg_moodtheme=59,
)

# help function
def convert_scores_to_outputs(exp: str, scores):
    output = torch.sigmoid(scores)
    return output

def aggregate_output_embedding(exp: str, emb):
    output = emb.mean(dim=1)
    return output


class BaseModel(pl.LightningModule):
    def __init__(self, exp: str):
        super().__init__()

        self.save_hyperparameters()
        self.exp = exp

        # To be set in configure_optimizers
        self.optimizer_cfg = {}
        self.scheduler_cfg = {}

        self.inference_segment_length = -1

        self.train_metrics = build_metrics(exp=exp, n_outputs=N_OUTPUTS[exp], split="train")
        self.val_metrics = build_metrics(exp=exp, n_outputs=N_OUTPUTS[exp], split="val")
        self.test_metrics = build_metrics(exp=exp, n_outputs=N_OUTPUTS[exp], split="test")
        
    def training_step(self, batch, batch_idx) -> Any:
        loss_dict, output = self.common_step(batch)

        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"])
        self.log_dict_prefix(loss_dict, "train")

        self.train_metrics.update(output, batch["y"])

        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx) -> Any:
        loss_dict, output = self.common_step(batch)

        self.log_dict_prefix(loss_dict, "val")

        self.val_metrics.update(output, batch["y"])

        return loss_dict["loss/total"]

    def test_step(self, batch, batch_idx) -> Any:
        loss_dict, output = self.common_step(batch, test=True)

        self.log_dict_prefix(loss_dict, "test")

        self.test_metrics.update(output, batch["y"])

    def common_step(self, batch, test=False):
        raise NotImplementedError

    def configure_optimizers(self) -> Any:
        raise NotImplementedError

    def segment_inference_input(self, x: torch.Tensor):
        # Discard the last segment
        N = x.shape[1] // self.inference_segment_length
        x = x[:, :self.inference_segment_length * N].reshape(N, self.inference_segment_length)
        return x

    def on_train_epoch_end(self) -> None:
        metrics_dict = self.train_metrics.compute()
        self.log_dict_prefix(metrics_dict, "train")
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        metrics_dict = self.val_metrics.compute()
        self.log_dict_prefix(metrics_dict, "val", prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        metrics_dict = self.test_metrics.compute()
        self.log_dict_prefix(metrics_dict, "test")
        # self.test_metrics.reset()

    def log_dict_prefix(self, d, prefix, **kwargs):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v, **kwargs)

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure
    ):
        # skip the first 500 steps
        if self.trainer.global_step < self.scheduler_cfg["warmup_steps"]:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.scheduler_cfg["warmup_steps"])
            lr = lr_scale * self.optimizer_cfg["args"]["lr"]
        else:
            step = self.trainer.global_step - self.scheduler_cfg["warmup_steps"]
            
            if step < self.scheduler_cfg["max_steps"]:
                lr = self.scheduler_cfg["lr_min"] + 0.5 * (self.optimizer_cfg["args"]["lr"] - self.scheduler_cfg["lr_min"]) * (1 + math.cos(math.pi * step / self.scheduler_cfg["max_steps"]))
            else:
                lr = self.scheduler_cfg["lr_min"]
            
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # update params
        optimizer.step(closure=optimizer_closure)