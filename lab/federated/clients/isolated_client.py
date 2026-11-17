from .basic_client import Client
from collections import OrderedDict
import torch


class IsolatedClient(Client):
    """
    Useless for know but will be usefull when some specific element of FedAvg
    wouldn't be needed to be in the client template class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def local_training(self, round_: int, global_weight=None) -> tuple[
        tuple[int, dict],
        tuple[int, dict]
    ]:
        """
        perform one round of training locally and return the new weight, local_dataset_size and computed_metrics

        :param global_weight:
        :param round_:

        :return:
            - new local weight learn
            - size of training data
            - ModuleDict of computed metrics

        :rtype tuple[OrderedDict[str, torch.Tensor], int, torch.nn.ModuleDict]
        """
        self.current_round = round_
        if self.steps_mode == 'epoch':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} epochs')
            self.trainer.partial_fit_epochs(epochs=self.max_steps)
        elif self.steps_mode == 'iter':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} iter')
            self.trainer.partial_fit_iterations(iterations=self.max_steps)

        train_metrics = self.model.train_metrics_epoch
        validation_metrics = self.processing_validation_metric(self.model.val_metrics_epoch)

        return ((len(self.datamodule.train_dataloader().dataset), train_metrics),
                (len(self.datamodule.train_dataloader().dataset), validation_metrics))
