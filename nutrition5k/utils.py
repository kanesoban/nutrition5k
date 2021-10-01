import argparse

import torch
from torchmetrics import Metric


def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', help='Name of the base config file without extension.', required=True)
    return parser.parse_args()


class Metrics:
    def __init__(self, task_list, device, prediction_threshold):
        self.task_list = task_list
        self.mean_absolute_errors = {}
        self.n5k_relative_mae = {}
        self.my_relative_mae = {}
        self.thresholded_accuracy = {}
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task] = MeanAbsoluteError(idx, device)
            self.n5k_relative_mae[task] = N5kRelativeMAE(idx, device)
            self.my_relative_mae[task] = MeanRelativeError(idx, device)
            self.thresholded_accuracy[task] = ThresholdedAccuracy(idx, device, prediction_threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task].update(preds, target)
            self.n5k_relative_mae[task].update(preds, target)
            self.my_relative_mae[task].update(preds, target)
            self.thresholded_accuracy[task].update(preds, target)

    def compute(self):
        metrics = {}
        for idx, task in enumerate(self.task_list):
            metrics['{} mean average error'.format(task)] = self.mean_absolute_errors[task].compute()
            metrics['{} n5k relative mean average error'.format(task)] = self.n5k_relative_mae[task].compute()
            #metrics['{} my relative mean average error'.format(task)] = self.my_relative_mae[task].compute()
            #metrics['{} thresholded accuracy'.format(task)] = self.thresholded_accuracy[task].compute()
        return metrics

    def reset(self):
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task].reset()
            self.n5k_relative_mae[task].reset()


class MeanAbsoluteError(Metric):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.add_state("abs_error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        task_preds = preds[:, self.task_idx]
        task_target = target[:, self.task_idx]

        n_samples = target.numel()
        dividends = torch.abs(task_target)
        dividends[dividends == 0] = 1
        abs_difference = torch.abs(task_preds - task_target)
        self.abs_error += torch.sum(abs_difference)
        self.total += n_samples

    def compute(self):
        return self.abs_error / self.total


class MeanRelativeError(Metric):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.add_state("relative_error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        task_preds = preds[:, self.task_idx]
        task_target = target[:, self.task_idx]

        n_samples = target.numel()
        dividends = torch.abs(task_target)
        dividends[dividends == 0] = 1
        abs_difference = torch.abs(task_preds - task_target)
        self.relative_error += torch.sum(abs_difference / dividends)
        self.total += n_samples

    def compute(self):
        return self.relative_error / self.total


class N5kRelativeMAE(Metric):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.task_mean = (255.0, 215.0, 12.7, 19.4, 18.0)[task_idx]
        self.add_state("error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n_samples = target.numel()
        self.error += torch.sum(torch.tensor(torch.abs(
            preds[:, self.task_idx] - target[:, self.task_idx]) / self.task_mean))
        self.total += n_samples

    def compute(self):
        return self.error / self.total


class ThresholdedAccuracy(Metric):
    def __init__(self, task_idx, device, prediction_threshold, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.prediction_threshold = prediction_threshold
        self.add_state("correct", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n_samples = target.numel()
        dividends = torch.abs(target[:, self.task_idx])
        dividends[dividends == 0] = 1
        self.correct += torch.sum(torch.tensor((torch.abs(
            preds[:, self.task_idx] - target[:, self.task_idx]) / dividends) < self.prediction_threshold))
        self.total += n_samples

    def compute(self):
        return self.correct / self.total
