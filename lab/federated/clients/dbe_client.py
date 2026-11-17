from collections import OrderedDict
from typing import Optional
import torch
from .basic_client import Client
from lab.models import DBEModel


class DBEClient(Client):

    def __init__(self, kappa, momentum, gamma, model, *args, **kwargs):
        model = DBEModel(base_model=model, kappa=kappa, momentum=momentum, gamma=gamma)
        super().__init__(model=model, *args, **kwargs)

    def set_global_mean(self, global_mean: torch.Tensor):
        self.model.global_mean = global_mean.to(self.model.device)

    def local_training(self, round_: int, global_weight: OrderedDict[str, torch.Tensor]) -> tuple[
        OrderedDict[str, torch.Tensor],
        torch.Tensor,
        tuple[int, dict],
        tuple[int, dict]
    ]:
        """
        perform one round of training locally and return the new weight, local_dataset_size and computed_metrics

        :param global_weight: global weight receive from server

        :type global_weight: OrderedDict[str, torch.Tensor]
        :type num_local_epoch: int

        :return:
            - new local weight learn
            - size of training data
            - ModuleDict of computed metrics

        :rtype tuple[OrderedDict[str, torch.Tensor], int, torch.nn.ModuleDict]
        """
        self.model.set_parameters(global_weight, round_)
        if self.steps_mode == 'epoch':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} epochs')
            self.trainer.partial_fit_epochs(epochs=self.max_steps)
        elif self.steps_mode == 'iter':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} iter')
            self.trainer.partial_fit_iterations(iterations=self.max_steps)


        train_metrics = self.model.train_metrics_epoch
        validation_metrics = self.processing_validation_metric(self.model.val_metrics_epoch)

        return (self.model.get_parameters(), self.model.get_running_mean(),
                (len(self.datamodule.train_dataloader().dataset), train_metrics),
                (len(self.datamodule.train_dataloader().dataset), validation_metrics))

    def compute_specific_mean(self, global_weight: Optional[OrderedDict[str, torch.Tensor]]):
        """
        perform one epoch before training to compute the client specific mean
        :param global_weight:
        :return:
        """
        # we might have a logging problem by doing a round before the real FL part
        if global_weight is not None:
            self.model.set_parameters(global_weight, 0)
        self.trainer.partial_fit_epochs(epochs=1)
        return self.model.get_running_mean(), len(self.datamodule.train_dataloader().dataset)
