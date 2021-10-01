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
        self.accuracies = {}
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task] = MeanAbsoluteError(idx, device)
            self.accuracies[task] = Accuracy(idx, device, prediction_threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task].update(preds, target)
            self.accuracies[task].update(preds, target)

    def compute(self):
        metrics = {}
        for idx, task in enumerate(self.task_list):
            metrics['{} mean average error'.format(task)] = self.mean_absolute_errors[task].compute()
            metrics['{} accuracy'.format(task)] = self.accuracies[task].compute()
        return metrics

    def reset(self):
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task].reset()
            self.accuracies[task].reset()


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


class Accuracy(Metric):
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
