from abc import ABC, abstractmethod
import lightning as L
import torch
from collections import OrderedDict


class BaseFedModule(L.LightningModule, ABC):
    def __init__(self):
        super().__init__()

    def set_parameters(self, parameters):
        """
        use to update local weight from the one received
        """
        self.load_state_dict(parameters)

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        get model parameter and put them on cpu before giving them.
        We make sure to detach the tensor
        :return:
        """
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.state_dict().items())
