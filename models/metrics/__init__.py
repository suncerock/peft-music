from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC, MultilabelF1Score, MulticlassAccuracy


def build_metrics(exp: str, n_outputs, split="train"):
    return MetricCollection({
        "mAP": MultilabelAveragePrecision(num_labels=n_outputs),
        "ROC-AUC": MultilabelAUROC(num_labels=n_outputs)
    })