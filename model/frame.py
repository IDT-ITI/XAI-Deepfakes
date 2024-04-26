import timm
from typing import Optional
from model.base_model import BaseModel
import torch
from model import utils


class BaseFrameModel(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = torch.cat([v[0] for k, v in batch.items() if v is not None])
            y = torch.cat([v[1] for k, v in batch.items() if v is not None])
        else:
            x, y = batch[0]

        # y = y.view(-1, 1)
        y = y.long()
        noise = None
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
        if isinstance(self.loss, utils.TradesLoss):
            loss_value, logits = self.loss(x, y, self.optimizers(), noise=noise)
        else:
            if noise is not None:
                x = x + noise
            logits = self.model(x)
            loss_value = self.loss(logits, y)  # type: ignore
        self.log(
            "train/loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        return {"loss": loss_value, "logits": logits, "y": y}

    def on_train_batch_end(self, output: dict, batch, batch_idx, dataloader_idx=None):
        preds = self.act(output["logits"])
        self.train_metrics.update(preds, output["y"])
        self.log_dict(
            self.train_metrics,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        # self.cm_train.update(preds, output["y"])

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        # y = y.view(-1, 1)
        y = y.long()
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
            x = x + noise

        logits = self.model(x)
        if isinstance(self.loss, utils.TradesLoss):
            loss_value = self.loss.natural_loss(logits, y)
        else:
            loss_value = self.loss(logits, y)  # type: ignore

        self.log(
            "val/loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        return {"loss": loss_value, "logits": logits, "y": y}

    def on_validation_batch_end(
        self, output: dict, batch, batch_idx, dataloader_idx=None
    ):
        preds = self.act(output["logits"])
        self.val_metrics.update(preds, output["y"])
        self.log_dict(
            self.val_metrics,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

    def on_test_start(self):
        self.test_step_outputs = {
            i: [] for i in range(len(self.trainer.datamodule.test_data))  # type: ignore
        }

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        # y = y.view(-1, 1)
        y = y.long()
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
            x = x + noise

        logits = self.model(x)
        loss_value = self.loss(logits, y)  # type: ignore

        preds = self.act(logits)

        if dataloader_idx is None:
            dataloader_idx = 0

            return {
                "preds": preds,
                "loss": loss_value,
                "logits": logits,
                "y": y,
            }

        self.test_step_outputs[dataloader_idx].append(  # type: ignore
            {"preds": preds, "loss": loss_value, "logits": logits, "y": y}
        )
        return {"preds": preds, "loss": loss_value, "logits": logits, "y": y}

    def on_test_batch_end(self, output: dict, batch, batch_idx, dataloader_idx=None):
        # loss = self.loss(output['logits'], output['y'])
        self.log(
            "test/loss",
            output["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )
        self.cm_test.update(output["preds"], output["y"])

    def on_test_epoch_end(self):
        for i, loader_output in self.test_step_outputs.items():  # type: ignore
            test_metrics = self.test_metrics.clone(
                prefix=f"test_{self.trainer.datamodule.test_data[i]}/"  # type: ignore
            )
            preds = torch.vstack([step["preds"] for step in loader_output])
            y = torch.cat([step["y"] for step in loader_output], dim=0)
            self.log_dict(
                test_metrics(preds, y),
                logger=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,  # type: ignore
            )

        self.test_step_outputs.clear()


class FrameModel(BaseFrameModel):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_path_rate: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model: torch.nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
            drop_path_rate=drop_path_rate,
        )
