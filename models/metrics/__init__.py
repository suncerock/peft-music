from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC, MultilabelF1Score, MulticlassAccuracy

from .key import KeyWeightedScore
from .tempo import TempoAcc

def build_metrics(exp: str, n_outputs, split="train"):
    if exp.startswith("mtat") or exp.startswith("mtg"):
        return MetricCollection({
            "mAP": MultilabelAveragePrecision(num_labels=n_outputs),
            "ROC-AUC": MultilabelAUROC(num_labels=n_outputs)
        })
    elif exp.endswith("key"):
        return MetricCollection({
            "acc": MulticlassAccuracy(num_classes=n_outputs),
            "weighted_score": KeyWeightedScore()
        })
    elif exp.endswith("tempo"):
        return MetricCollection({
            "acc_1": TempoAcc(acc_type=1),
            # "acc_2": TempoAcc(acc_type=2),  # There is a bug when using acc 1 and acc2 together
        })    