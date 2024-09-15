import torch
from torchmetrics import Metric

class KeyWeightedScore(Metric):
    NUM_KEYS = 12
    MAJOR_MODE = 0
    MINOR_MODE = 1
    ALL_CLASSES = [
        "C major", "Db major", "D major", "Eb major", "E major", "F major", "Gb major", "G major", "Ab major", "A major", "Bb major", "B major",
        "C minor", "Db minor", "D minor", "Eb minor", "E minor", "F minor", "Gb minor", "G minor", "Ab minor", "A minor", "Bb minor", "B minor"
    ]

    def __init__(self):
        super().__init__()

        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target) -> None:
        preds = torch.argmax(preds, dim=-1)
        
        estimated_key = preds % KeyWeightedScore.NUM_KEYS
        reference_key = target % KeyWeightedScore.NUM_KEYS
        estimated_mode = preds // KeyWeightedScore.NUM_KEYS
        reference_mode = target // KeyWeightedScore.NUM_KEYS

        self.correct += 1.0 * torch.count_nonzero((reference_key == estimated_key) * (reference_mode == estimated_mode))
        self.correct += 0.5 * torch.count_nonzero((estimated_mode == reference_mode) *\
                                                    ((estimated_key - reference_key) % 12 == 7))
        self.correct += 0.5 * torch.count_nonzero((estimated_mode == reference_mode) *\
                                                    ((estimated_key - reference_key) % 12 == 5))
        self.correct += 0.3 * torch.count_nonzero((estimated_mode != reference_mode) * (reference_mode == KeyWeightedScore.MAJOR_MODE) *\
                                                    (estimated_key - reference_key) % 12 == 9)
        self.correct += 0.3 * torch.count_nonzero((estimated_mode != reference_mode) * (reference_mode == KeyWeightedScore.MINOR_MODE) *\
                                                    (estimated_key - reference_key) % 12 == 3)
        self.correct += 0.2 * torch.count_nonzero((estimated_mode != reference_mode) * (estimated_key == reference_key))

        self.total += len(target)

    def compute(self):
        return self.correct / self.total

