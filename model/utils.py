import io
import itertools
from lightning.pytorch.callbacks import Callback
import subprocess
from matplotlib import pyplot as plt
from torchmetrics import Metric, Recall
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from PIL import Image


class FalsePositiveRate(Metric):
    def __init__(self, threshold=0.5, eps=1e-8, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.eps = eps

        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_binary = (preds >= self.threshold).int()
        self.false_positives += torch.sum((preds_binary == 1) & (target == 0))
        self.negatives += torch.sum(target == 0)

    def compute(self):
        return self.false_positives.float() / (self.negatives.float() + self.eps)


class FalseNegativeRate(Metric):
    def __init__(self, threshold=0.5, eps=1e-8, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.eps = eps

        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("positives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_binary = (preds >= self.threshold).int()
        self.false_negatives += torch.sum((preds_binary == 0) & (target == 1))
        self.positives += torch.sum(target == 1)

    def compute(self):
        return self.false_negatives.float() / (self.positives.float() + self.eps)


class BalancedAccuracy(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: int = 2,
        average: str = "macro",
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.recall = Recall(
            num_classes=num_classes, average=average, task="multiclass"
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert probabilities to binary predictions
        preds_binary = (preds >= self.threshold).int()
        self.recall.update(preds_binary, target)

    def compute(self):
        # Compute the final balanced accuracy (recall) score
        return self.recall.compute()

    def reset(self):
        # Reset the recall metric
        self.recall.reset()


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix with actual numbers.

    Args:
      cm (Tensor): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    cm = cm.cpu().numpy()  # Convert to numpy array
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f} ({cm[i, j]})",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def convert_figure_to_tensor(figure):
    """
    Converts a matplotlib figure to a 3D tensor normalized to [0, 1].
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).float() / 255.0  # Normalize to [0, 1]
    return image_tensor
