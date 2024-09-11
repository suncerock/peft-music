from typing import Dict

import torch.nn as nn


class TaggingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target.float())


ALL_LOSSES: Dict[str, nn.Module] = dict(
    mtat=TaggingLoss,

    mtg_top50=TaggingLoss,
    mtg_genre=TaggingLoss,
    mtg_instrument=TaggingLoss,
    mtg_moodtheme=TaggingLoss,
)