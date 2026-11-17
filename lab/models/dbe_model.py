import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from typing import Sequence, Union
from collections import OrderedDict
from torch import nn
from lightning import LightningModule
from .base_fed_model import BaseFedModel

class PRBM(LightningModule):
    """
    Personalized Representation Bias Memory
    Is a local parameter and thus shouldn't be sent to global server
    """
    def __init__(self, dim: Union[Sequence, int]):
        super().__init__()
        self.learned_bias = nn.Parameter(torch.zeros(dim))  # called client_mean in the original MBE implementation

    def forward(self, x):
        return x + self.learned_bias

def MR(global_mean: torch.Tensor, client_encoded_features: torch.Tensor):
    """
    Compute the Mean Regularization of DBE paper

    :param global_mean: <-> mean z^g (in DBE paper)
    :param client_encoded_features:  <-> mean z_i^g (in DBE paper)
    :return:
    """
    assert global_mean.shape == client_encoded_features.shape
    return torch.mean(0.5 * (client_encoded_features - global_mean) ** 2)


class DBEModel(BaseFedModel):
    """
    act as a wrapper of model to which the DBE mechanism is applied
    """
    def __init__(self, base_model: BaseFedModel, kappa, momentum, gamma):

        # calling the parent constructor (BaseFedModel) and mapping its parameters from the provided base_model
        # as DBE play the role of
        super().__init__(loss=base_model.loss,
                         train_metrics=base_model.train_metrics_dict,
                         validation_metrics=base_model.validation_metrics_dict,
                         test_metrics=base_model.test_metrics_dict,
                         monitor=base_model.monitor,
                         monitor_objective=base_model.monitor_objective,
                         gradient_logging=base_model.gradient_logging,
                         weighted_loss=base_model.weighted_loss)
        # check that the provided model implement encode and decode function
        # as the DBE module is taking place between the both.
        assert hasattr(base_model, 'encode') and callable(base_model.encode)
        assert hasattr(base_model, 'decode') and callable(base_model.decode)
        assert hasattr(base_model, 'get_hyperparameters') and callable(base_model.get_hyperparameters)
        assert hasattr(base_model, 'latent_space') and isinstance(base_model.latent_space, Union[Sequence, int])

        self.base_model = base_model

        # retrieve the hyperparameters of the wrapped model
        self.learning_rate = base_model.learning_rate
        self.weight_decay = base_model.weight_decay
        self.gradient_clip_val = base_model.gradient_clip_val

        # DBE use two optimizer and thus required that we manually define the optimization
        # We have one optimizer for the provided base_model parameters (encoder + decoder)
        # We have one optimizer for the PRBM (client bias).
        self.automatic_optimization = False

        # initiate as None as DBE must do one epoch w/o MR et PRBM.
        self.global_mean = None

        self.prbm = PRBM(base_model.latent_space)

        self.kappa = kappa  # hyperparameter to regulate the divergence between mean z_i^g to mean z_g
        self.momentum = momentum  # hyperparameter to controls how much new data influences the EMA
        self.gamma = gamma  # gamma learning rate decay

        self.global_mean = None  # mean z^g
        # EMA moving average compute in DBE implementation using a momentum hyperparameter
        self.running_mean = torch.zeros(base_model.latent_space)

    def set_global_mean(self, global_mean):
        self.global_mean = global_mean

    def get_running_mean(self):
        return self.running_mean.detach().cpu().clone()

    """
    used for inference/prediction
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base_model.encode(x)
        if self.global_mean is not None: # equiv to say that we are not in the init epoch and thus we use DBE mechanism
            z = self.prbm(z)
        y = self.base_model.decode(z)
        return y

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        get model parameter and put them on cpu before giving them.
        We make sure to detach the tensor
        :return:
        """
        return self.base_model.get_parameters()
        # return OrderedDict((k, v.detach().cpu().clone()) for k, v in self.base_model.state_dict().items())

    def set_parameters(self, parameters, current_round: int):
        """
        use to update local weight from the one received
        """
        self.base_model.set_parameters(parameters, current_round)

    def on_train_epoch_start(self):
        # reset the running_mean to zero before each training epoch
        # as it supposed to contain the moving average of the current epoch
        self.running_mean.zero_()
        self.running_mean = self.running_mean.to(self.device)

    def training_step(self, batch, batch_idx):
        model_optimizer, prbm_optimizer = self.optimizers()
        # model_optimizer = self.optimizers()
        model_lr_scheduler = self.lr_schedulers()  # only one scheduler for the model
        x, y = batch
        z = self.base_model.encode(x)
        running_mean = torch.mean(z, dim=0)
        # self.running_mean.to(self.device)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * running_mean  # EMA moving average

        if self.global_mean is not None:
            z = self.prbm(z)  # z_g + z_p
            y_hat = self.base_model.decode(z)
            loss = self.loss(y_hat, y)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            mr_reg = MR(self.global_mean, self.running_mean)
            loss = loss + self.kappa * mr_reg
            self.log("train/loss_dbe", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train/mr', mr_reg, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:  # training step w/o DBE (PRBM + MR)
            y_hat = self.base_model.decode(z)
            loss = self.loss(y_hat, y)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Manual optimization
        prbm_optimizer.zero_grad()
        model_optimizer.zero_grad()
        self.manual_backward(loss)

        self.clip_gradients(
            model_optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm"
        )
        model_optimizer.step()

        self.clip_gradients(
            prbm_optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm"
        )
        prbm_optimizer.step()
        # model_lr_scheduler.step()
        # detach the current running_mean from the computational graph as the current value have already been used to gradient update
        self.running_mean.detach_()

        self.log_images(batch)
        for key in self.train_metrics:
            metric = self.train_metrics[key](y_hat, y)
            self.log(f"train/{key}", metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss


    def on_train_epoch_end(self) -> None:
        """
        only save the metrics of last epoch
        :return:
        """
        self.train_metrics_epoch['train/loss'] = self.trainer.callback_metrics.get("train/loss").item()
        if "train/loss_dbe" in self.trainer.callback_metrics:
            self.train_metrics_epoch['train/loss_dbe'] = self.trainer.callback_metrics.get("train/loss_dbe").item()
        for key in self.train_metrics:
            self.train_metrics_epoch[f"train/{key}"] = self.trainer.callback_metrics.get(f"train/{key}").item()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(list(self.base_model.parameters()) + list(self.prbm.parameters()), lr=self.learning_rate)
        model_optimizer = torch.optim.SGD(self.base_model.parameters(), lr=self.learning_rate)
        # lr_scheduler_model = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=self.gamma)

        prbm_optimizer = torch.optim.SGD(self.prbm.parameters(), lr=self.learning_rate)
        # return {"optimizer": model_optimizer, "lr_scheduler": lr_scheduler_model}, {"optimizer": prbm_optimizer}
        return model_optimizer, prbm_optimizer
        # return optimizer
