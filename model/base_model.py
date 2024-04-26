from typing import Dict, Optional
import numpy as np
from typing import Any, Dict, List, Optional
from lightning import LightningModule
import torch
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
    Specificity,
    ConfusionMatrix,
)
import model.utils as utils

class BaseModel(LightningModule):
    def __init__(
        self,
        task: str = "binary",
        num_classes: int = 1,
        adv_attacks: Optional[List[str]] = None,
        test_after_fit: bool = False,
        aggregate_predict_results: bool = False,
        use_pos_weight: bool = False,
        loss_str: str = "bce_with_logits",
        natural_loss_str: str = "bce_with_logits",
        criterion_kl: str = "kldiv",
        distance: str = "l_inf",
        perturb_steps: int = 10,
        step_size: float = 0.003,
        epsilon: float = 0.031,
        beta: float = 6.0,
        random_smoothing: bool = False,
        random_smoothing_noise_sd: float = 0.12,
        gamma: float = 2.0,
        alpha: float = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.task = task
        self.num_classes = num_classes
        assert self.task in [
            "binary",
            "multiclass",
        ], "task must be binary or multiclass"

        self.adv_attacks = adv_attacks or []
        self.test_after_fit = test_after_fit
        self.aggregate_predict_results = aggregate_predict_results
        self.use_pos_weight = use_pos_weight
        self.loss_str = loss_str
        self.natural_loss_str = natural_loss_str
        self.criterion_kl = criterion_kl
        self.distance = distance
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.beta = beta
        self.random_smoothing = random_smoothing
        self.random_smoothing_noise_sd = random_smoothing_noise_sd
        self.gamma = gamma
        self.alpha = alpha

        self.test_step_outputs: Dict[int, List[dict]] = {}

        # activation function
        if self.task == "binary":
            self.act = torch.sigmoid
        elif self.task == "multiclass":
            self.act = lambda x: torch.softmax(x, dim=1)

        if self.task == "binary":
            self.metrics = MetricCollection(
                [
                    Accuracy(task="binary"),
                    utils.BalancedAccuracy(
                        threshold=0.5, num_classes=2, average="macro"
                    ),
                    Recall(task="binary"),
                    AUROC(task="binary"),
                    F1Score(task="binary"),
                    Precision(task="binary"),
                    Specificity(task="binary"),
                    utils.FalsePositiveRate(),
                    utils.FalseNegativeRate(),
                ]
            )
            self.cm_train = ConfusionMatrix(num_classes=2, task="binary")
            self.cm_val = ConfusionMatrix(num_classes=2, task="binary")
            self.cm_test = ConfusionMatrix(num_classes=2, task="binary")

        elif self.task == "multiclass":
            self.metrics = MetricCollection(
                [
                    Accuracy(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                        multidim_average="global",
                    ),
                    Precision(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                    ),
                    Recall(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                    ),
                    F1Score(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                    ),
                    AUROC(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                    ),
                    AveragePrecision(
                        task="multiclass",
                        num_classes=self.num_classes,
                        average="weighted",
                    ),
                ]
            )
            self.cm_train = ConfusionMatrix(
                num_classes=self.num_classes, task="multiclass"
            )
            self.cm_val = ConfusionMatrix(
                num_classes=self.num_classes, task="multiclass"
            )
            self.cm_test = ConfusionMatrix(
                num_classes=self.num_classes, task="multiclass"
            )

        self.train_metrics = self.metrics.clone(prefix="train/")
        self.val_metrics = self.metrics.clone(prefix="val/")
        self.test_metrics = self.metrics.clone(prefix="test/")

    def setup(self, stage: str) -> None:
        if self.loss_str == "bce_with_logits":
            if self.use_pos_weight and stage == "fit":
                dataset_name = self.trainer.datamodule.train_data[0]  # type: ignore
                n_pos_samples = self.trainer.datamodule.datasets[dataset_name][  # type: ignore
                    "train"
                ].n_pos_labels
                n_neg_samples = (
                    len(
                        self.trainer.datamodule.datasets[dataset_name]["train"]  # type: ignore
                    )
                    - n_pos_samples
                )
                pos_weight = n_neg_samples / n_pos_samples
                print(f"Using pos_weight {pos_weight}")
                self.loss = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weight)
                )
                self.pos_weight = pos_weight
            else:
                self.loss = torch.nn.BCEWithLogitsLoss()
        elif self.loss_str == "focal":
            self.loss = utils.FocalLoss(alpha=self.alpha, gamma=self.gamma)
        elif self.loss_str == "trades":
            if self.natural_loss_str == "bce_with_logits":
                self.natural_loss = torch.nn.BCEWithLogitsLoss()
            else:
                raise NotImplementedError(
                    f"natural loss {self.natural_loss_str} not implemented"
                    f" for trades loss"
                )
            if self.criterion_kl == "kldiv":
                self.criterion = torch.nn.KLDivLoss(size_average=False)
            else:
                raise NotImplementedError(
                    f"criterion {self.criterion_kl} not implemented" f" for trades loss"
                )
            self.loss = utils.TradesLoss(
                self.model,  # type: ignore
                self.natural_loss,
                self.criterion,
                distance=self.distance,  # type: ignore
                perturb_steps=self.perturb_steps,  # type: ignore
                step_size=self.step_size,  # type: ignore
                epsilon=self.epsilon,  # type: ignore
                beta=self.beta,  # type: ignore
                random_smoothing=self.random_smoothing,  # type: ignore
            )
        elif self.loss_str == "ce":
            if self.use_pos_weight and stage == "fit":
                dataset_name = self.trainer.datamodule.train_data[0]
                class_counts = self.trainer.datamodule.datasets[dataset_name][
                    "train"
                ].class_counts
                class_ratios = np.array(list(class_counts.values())) / len(
                    self.trainer.datamodule.datasets[dataset_name][
                        "train"
                    ].labeled_video_paths
                )

                print(f"Class ratios are: {class_ratios}")
                # calcuate inverse of class frequencies
                weights = 1.0 / class_ratios
                # Normalize weights so that the smallest weight is 1.0
                weights = weights / weights.min()
                print(f"Using class weights: {weights}")

                self.loss = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(weights)
                )

            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"loss {self.loss_str} not implemented")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # remove pos_weight from checkpoint when loading for testing / inference
        checkpoint["state_dict"].pop("loss.pos_weight", None)

        return super().on_load_checkpoint(checkpoint)

    def forward(self, x):
        return self.act(self.model(x))  # type: ignore

    def predict_step(self, batch, batch_idx=None):
        return self(batch)

    def _normalize_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
