import numpy as np

import torch
from torchmetrics import Metric


class TempoAcc(Metric):
    """
    Implementation from
    https://tempobeatdownbeat.github.io/tutorial/intro.html#
    """
    def __init__(self, acc_type=1) -> None:
        super().__init__()

        self.acc_type = acc_type

        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        target = torch.argmax(target, dim=-1).detach().cpu().numpy().copy()

        preds = preds.float().detach().cpu().numpy().copy()
        output = np.array([self.detect_tempo(pred) for pred in preds])

        if self.acc_type == 1:
            errors = np.abs(1 - (output / target))
            # correctly identified annotation tempi
            self.correct += np.sum(errors <= 0.04)
            self.total += len(output)
        else:
            target = np.vstack([target, target * 2, target / 2, target * 3, target / 3])
            errors = np.abs(1 - output / target)
            self.correct += np.sum((errors <= 0.04).any(axis=0))
            self.total += len(output)

    def compute(self):
        return self.correct / self.total

    def detect_tempo(self, bins, min_bpm=10, hist_smooth=11):
        from scipy.interpolate import interp1d
        from scipy.signal import argrelmax
        min_bpm = int(np.floor(min_bpm))
        tempi = np.arange(min_bpm, len(bins))
        bins = bins[min_bpm:]
        # smooth histogram bins
        if hist_smooth > 0:
            kernel = np.hamming(hist_smooth)
            bins = np.convolve(bins, kernel, 'same')
        # create interpolation function
        interpolation_fn = interp1d(tempi, bins, 'quadratic')
        # generate new intervals with 1000x the resolution
        tempi = np.arange(tempi[0], tempi[-1], 0.001)
        # apply quadratic interpolation
        bins = interpolation_fn(tempi)
        peaks = argrelmax(bins, mode='wrap')[0]
        if len(peaks) == 0:
            # no peaks, no tempo
            tempi = np.array([], ndmin=2)
        elif len(peaks) == 1:
            # report only the strongest tempo
            tempi = np.array([tempi[peaks[0]], 1.0])
        else:
            # sort the peaks in descending order of bin heights
            sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
            # normalize their strengths
            strengths = bins[sorted_peaks]
            strengths /= np.sum(strengths)
            # return the tempi and their normalized strengths
            tempi = np.array(list(zip(tempi[sorted_peaks], strengths)))
        return tempi[0, 0]