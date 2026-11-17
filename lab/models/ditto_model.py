import copy
from collections import OrderedDict
from typing import Optional

import torch

from .base_fed_model import BaseFedModel


def _extract_optimizer(configured):
    """Normalize the various shapes LightningModule.configure_optimizers() can return down to a single optimizer."""
    if isinstance(configured, dict):
        return configured['optimizer']
    if isinstance(configured, (list, tuple)):
        return configured[0]
    return configured


class DittoModel(BaseFedModel):
    """
    Wrapper implementing Ditto (Li et al. 2021 - "Ditto: Fair and Robust Federated Learning through Personalization").

    Holds two copies of the base architecture:
    - base_model: trained locally like a plain FedAvg client, its weights are the ones aggregated by the server.
    - personalized_model: persists across rounds, never communicated, trained with a proximal term
      regularizing it towards the global weights received at the start of the round.
    """

    def __init__(self, base_model: BaseFedModel, lambda_reg: float):
        super().__init__(loss=base_model.loss,
                         train_metrics=base_model.train_metrics_dict,
                         validation_metrics=base_model.validation_metrics_dict,
                         test_metrics=base_model.test_metrics_dict,
                         monitor=base_model.monitor,
                         monitor_objective=base_model.monitor_objective,
                         gradient_logging=base_model.gradient_logging,
                         weighted_loss=base_model.weighted_loss)

        self.base_model = base_model
        self.personalized_model = copy.deepcopy(base_model)

        self.lambda_reg = lambda_reg

        # Ditto trains base_model and personalized_model with two independent forward/backward passes per batch
        self.automatic_optimization = False

        # frozen snapshot of the global weights received at the start of the current round (w_t),
        # used as the anchor of the proximal term for the whole round
        self._anchor_state_dict: Optional[OrderedDict[str, torch.Tensor]] = None

    def set_parameters(self, parameters, current_round: int):
        """
        Load the received global weight into the base_model and freeze it as the proximal anchor for this round.
        """
        self.base_model.set_parameters(parameters, current_round)
        self.current_round = current_round
        self._anchor_state_dict = OrderedDict((k, v.detach().clone()) for k, v in parameters.items())

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        Only base_model's weights are ever communicated, personalized_model stays local.
        """
        return self.base_model.get_parameters()

    def set_class_weights(self, class_weights: torch.Tensor) -> bool:
        """
        base_model and personalized_model hold independent loss objects (deepcopy), both must be updated.
        """
        base_ok = self.base_model.set_class_weights(class_weights)
        personalized_ok = self.personalized_model.set_class_weights(class_weights)
        return base_ok and personalized_ok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference/prediction. Client-local train/val/test metrics are reported against the
        personalized model, as this is the model Ditto's personalization is meant to be evaluated on.
        """
        return self.personalized_model(x)

    def training_step(self, batch, batch_idx):
        global_optimizer, personalized_optimizer = self.optimizers()
        x, y = batch

        # plain local update of the model that gets aggregated (identical in spirit to a FedAvg client)
        y_hat_global = self.base_model(x)
        loss_global = self.base_model.loss(y_hat_global, y)
        global_optimizer.zero_grad()
        self.manual_backward(loss_global)
        global_optimizer.step()

        # personalized update, regularized towards the frozen anchor w_t
        y_hat_personal = self.personalized_model(x)
        loss_task = self.personalized_model.loss(y_hat_personal, y)
        prox_reg = sum(
            torch.sum((p - self._anchor_state_dict[name].to(p.device)) ** 2)
            for name, p in self.personalized_model.named_parameters()
            if name in self._anchor_state_dict
        )
        loss_personal = loss_task + (self.lambda_reg / 2) * prox_reg
        personalized_optimizer.zero_grad()
        self.manual_backward(loss_personal)
        personalized_optimizer.step()

        self.log_images(batch)
        self.log("train/loss", loss_personal, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_global", loss_global, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/prox_reg", prox_reg, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        for key in self.train_metrics:
            metric = self.train_metrics[key](y_hat_personal, y)
            self.log(f"train/{key}", metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss_personal

    def on_train_epoch_end(self) -> None:
        self.train_metrics_epoch['train/loss'] = self.trainer.callback_metrics.get("train/loss").item()
        self.train_metrics_epoch['train/loss_global'] = self.trainer.callback_metrics.get("train/loss_global").item()
        self.train_metrics_epoch['train/prox_reg'] = self.trainer.callback_metrics.get("train/prox_reg").item()
        for key in self.train_metrics:
            self.train_metrics_epoch[f"train/{key}"] = self.trainer.callback_metrics.get(f"train/{key}").item()

    def configure_optimizers(self):
        global_optimizer = _extract_optimizer(self.base_model.configure_optimizers())
        personalized_optimizer = _extract_optimizer(self.personalized_model.configure_optimizers())
        return global_optimizer, personalized_optimizer
