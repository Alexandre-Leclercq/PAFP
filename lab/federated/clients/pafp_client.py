from collections import OrderedDict
from typing import Optional, Literal
import torch
from .basic_client import Client
from lab.models import DBEModel


class PAFPClient(Client):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def local_training(self, round_: int, round_type: Optional[Literal['opt_global', 'opt_local']],
                       global_weight: OrderedDict[str, torch.Tensor], global_statistics: list[dict]) -> tuple[
        OrderedDict[str, torch.Tensor],
        torch.Tensor,
        tuple[int, dict],
        tuple[int, dict]
    ]:
        """
        perform one round of training locally and return the new weight, local_dataset_size and computed_metrics

        :param round_: the round number
        :param round_type: the round type Optional[Literal['opt_global', 'opt_local']],
        :param global_weight: global weight receive from server
        :param global_statistics: global statistics receive from server

        :type round_: int
        :type global_weight: OrderedDict[str, torch.Tensor]
        :type global_statistics: list[dict]

        :return:
            - new local weight learn
            - size of training data
            - ModuleDict of computed metrics

        :rtype tuple[OrderedDict[str, torch.Tensor], torch.Tensor, tuple[int, dict], tuple[int, dict]]
        """
        self.model.set_parameters(global_weight, round_)
        if global_statistics is not None: # global_statistics is None for the first round
            self.model.set_aligner_global_statistics(global_statistics)
        self.model.opt_mode = round_type
        if self.steps_mode == 'epoch':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} epochs')
            self.trainer.partial_fit_epochs(epochs=self.max_steps)
        elif self.steps_mode == 'iter':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} iter')
            self.trainer.partial_fit_iterations(iterations=self.max_steps)

        train_metrics = self.model.train_metrics_epoch
        validation_metrics = self.processing_validation_metric(self.model.val_metrics_epoch)

        return (self.model.get_parameters(), self.model.get_aligner_global_statistics(),
                (len(self.datamodule.train_dataloader().dataset), train_metrics),
                (len(self.datamodule.train_dataloader().dataset), validation_metrics))
