import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Sequence
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import torchvision
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lab.utils.logging import model_log_pretty_class_weight
from lab.utils.loss import FocalLoss
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.trainer.states import TrainerFn
import lightning as L
import torch
from torch import nn
from collections import OrderedDict
from torchmetrics import MeanMetric


class BaseFedModel(L.LightningModule, ABC):
    def __init__(self, loss,
                 train_metrics,
                 validation_metrics,
                 test_metrics,
                 monitor,
                 monitor_objective: str,
                 gradient_logging,
                 weighted_loss = False,
                 local_params_keys: Sequence[Optional['str']] = ()):
        super().__init__()

        self.loss = loss
        # counter of current round
        # stay None if model not used in federated settings
        # only used to have pretty logging per round
        self.current_round = None

        # state variable use to log the first training batch once.
        self.has_logged_images = False
        self.gradient_logging = gradient_logging
        self.weighted_loss = weighted_loss
        self.local_params_keys = tuple(local_params_keys)

        # store metrics in dict of metric as simple dict doesn't allow for dynamic device setting

        # self.validation_metrics = nn.ModuleDict(validation_metrics)
        # self.test_metrics = nn.ModuleDict(test_metrics)
        self.train_metrics_dict = train_metrics
        self.validation_metrics_dict = validation_metrics
        self.test_metrics_dict = test_metrics

        self.monitor = monitor
        self.monitor_objective = monitor_objective

        # nn.ModuleDict containing the metrics object
        self.train_metrics = nn.ModuleDict(train_metrics)
        # will be dynamically created using a hook in case multiple dataloaders are provided
        self.validation_metrics = None
        # will be dynamically created using a hook in case multiple dataloaders are provided
        self.test_metrics = None

        self.size_validation_dataloaders: list = []
        self.size_test_dataloaders: list = []

        # dict that will contain all the metrics computed within one epoch (train + val)
        self.train_metrics_epoch: dict = {}
        self.val_metrics_epoch: dict = {}
        self.test_metrics_epoch: dict = {}

    def set_parameters(self, parameters, current_round: int):
        """
        use to update local weight from the one received
        """
        self.current_round = current_round
        self.load_state_dict(parameters, strict=False)

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        get model parameter and put them on cpu before giving them.
        We make sure to detach the tensor.
        Keep local the params mention in local_params_keys.
        :return:
        """
        # list the params to drop (not communicate)
        model_params = OrderedDict(
            (k, v.detach().cpu().clone())
            for k, v in self.state_dict().items()
            if not k.startswith(self.local_params_keys) # exclude parameters having the mentioned prefix name
        )
        return model_params

    def log_images(self, batch):
        if not self.has_logged_images:
            images, labels = batch
            grid = torchvision.utils.make_grid(images)
            # print(f'images space, min:{images.min()}, max:{images.max()}')
            for logger in self.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.add_image('Training batch', grid, global_step=0)
            self.has_logged_images = True

    def set_class_weights(self, class_weights: torch.Tensor) -> bool:
        """
        try to assign class_weights to the loss and return a boolean to indicate the state of success
        :param class_weights:
        :return:
        """
        print('try to set class_weights to : ', class_weights)
        if self.weighted_loss:
            if isinstance(self.loss, torch.nn.CrossEntropyLoss):
                self.loss.weight = class_weights.to(self.device)
                print('class weight assign to the loss')
                return True
            elif isinstance(self.loss, FocalLoss) and hasattr(self.loss, 'alpha'):
                self.loss.alpha = class_weights.to(self.device)
                print('class weight assign to the loss')
                return True
            else:
                logging.warning(f'{self.loss.__class__.__name__} is not a weighted loss')
        else:
            print(f'no class weight')
        return False

    def setup(self, stage: str):
        # Log the shared parameters
        if self.logger:
            # Access the TensorBoard SummaryWriter
            writer = self.logger.experiment
            model_params = self.get_parameters()
            log_text = ''
            for k in model_params.keys():
                log_text += f'{k}\t'
            writer.add_text('shared_param', log_text)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for a specific layer
        if self.gradient_logging:
            norms_model = {k:v for k, v in grad_norm(self, norm_type=2).items()}
            self.log_dict(norms_model)

    def on_train_epoch_end(self) -> None:
        """
        only save the metrics of last epoch
        :return:
        """
        self.train_metrics_epoch['train/loss'] = self.trainer.callback_metrics.get("train/loss").item()
        for key in self.train_metrics:
            self.train_metrics_epoch[f"train/{key}"] = self.trainer.callback_metrics.get(f"train/{key}").item()


    def on_validation_epoch_start(self) -> None:
        """
        use to init the validation_metrics object which will contain the metrics value computed
        on the validation epoch.
        """
        if self.trainer.datamodule is None:
            self.size_validation_dataloaders = [len(dataloader.dataset) for dataloader in self.trainer.val_dataloaders]
        else:
            self.size_validation_dataloaders = None
        val_metric_dict = {}
        if self.size_validation_dataloaders is None: # local validation
            val_metric_dict[f'loss'] = MeanMetric()
            for metric_name, metric_object in self.validation_metrics_dict.items():
                val_metric_dict[f'{metric_name}'] = metric_object.clone()
        else: # validation using multiple val_dataloader
            for i in range(len(self.size_validation_dataloaders)):
                val_metric_dict[f'loss_{i}'] = MeanMetric()
                for metric_name, metric_object in self.validation_metrics_dict.items():
                    val_metric_dict[f'{metric_name}_{i}'] = metric_object.clone()
        self.validation_metrics = nn.ModuleDict(val_metric_dict).to(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if self.size_validation_dataloaders is None: # local validation case
            self.validation_metrics[f'loss'].update(loss, weight=x.size(0))
            for key in self.validation_metrics_dict:
                self.validation_metrics[f'{key}'].update(y_hat, y)
        else: # validation using multiple val_dataloader
            self.validation_metrics[f'loss_{dataloader_idx}'].update(loss, weight=x.size(0))
            for key in self.validation_metrics_dict:
                self.validation_metrics[f'{key}_{dataloader_idx}'].update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        """
        only save the metrics of last epoch
        :return:
        """
        # Loop through each dataloader's metrics
        dict_log = {}
        if self.size_validation_dataloaders is None: # local validation
            validation_epoch_loss = self.validation_metrics[f'loss'].compute().item()
            dict_log[f'val/loss'] = validation_epoch_loss
            self.val_metrics_epoch[f'val/loss'] = validation_epoch_loss
            for key in self.validation_metrics_dict:
                metric_epoch_loss = self.validation_metrics[f'{key}'].compute().item()
                self.val_metrics_epoch[f'val/{key}'] = metric_epoch_loss
                dict_log[f'val/{key}'] = metric_epoch_loss
        else: # validation using multiple val_dataloader
            for i in range(len(self.size_validation_dataloaders)):
                validation_epoch_loss = self.validation_metrics[f'loss_{i}'].compute().item()
                dict_log[f'val/loss_{i}'] = validation_epoch_loss
                self.val_metrics_epoch[f'val/loss_{i}'] = validation_epoch_loss
                for key in self.validation_metrics_dict:
                    metric_epoch_loss = self.validation_metrics[f'{key}_{i}'].compute().item()
                    self.val_metrics_epoch[f'val/{key}_{i}'] = metric_epoch_loss
                    dict_log[f'val/{key}_{i}'] = metric_epoch_loss
        if self.trainer.state.fn == TrainerFn.FITTING: # when validation is performed by train.fit
            self.log_dict(dict_log, add_dataloader_idx=False, on_step=False, on_epoch=True)
        elif self.trainer.state.fn == TrainerFn.VALIDATING: # when validation is performed by train.validate
            # in this case we need to log by training round as it's not the validation performed in parallel of training
            for logger in self.trainer.loggers:
                logger.log_metrics(dict_log, step=self.current_round)
        else:
            raise ValueError('Unknown trainer state')

    def on_test_epoch_start(self) -> None:
        """
        use to init the test_metrics object which will contain the metrics value computed
        on the test epoch.
        """
        if getattr(self.trainer, "test_dataloaders", None) is not None:
            dataloaders = self.trainer.test_dataloaders
            if isinstance(dataloaders, list):
                self.size_test_dataloaders = [len(dataloader.dataset) for dataloader in dataloaders]
            else:
                self.size_test_dataloaders = None
        else:
            raise ValueError("No test dataloaders were provided to the Trainer! ")
        # if self.trainer.datamodule is None and type(self.trainer.test_dataloaders) == 'list':
        #     self.size_test_dataloaders = [len(dataloader.dataset) for dataloader in self.trainer.test_dataloaders]
        # else:
        #     self.size_test_dataloaders = None
        test_metric_dict = {}
        if self.size_test_dataloaders is None: # local test
            test_metric_dict[f'loss'] = MeanMetric()
            for metric_name, metric_object in self.test_metrics_dict.items():
                test_metric_dict[f'{metric_name}'] = metric_object.clone()
        else: # testing using multiple test_dataloader
            for i in range(len(self.size_test_dataloaders)):
                test_metric_dict[f'loss_{i}'] = MeanMetric()
                for metric_name, metric_object in self.test_metrics_dict.items():
                    test_metric_dict[f'{metric_name}_{i}'] = metric_object.clone()
        self.test_metrics = nn.ModuleDict(test_metric_dict).to(self.device)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        if self.size_test_dataloaders is None:  # local validation case
            self.test_metrics[f'loss'].update(loss, weight=x.size(0))
            for key in self.test_metrics_dict:
                self.test_metrics[f'{key}'].update(y_hat, y)
        else: # validation using multiple val_dataloader
            self.test_metrics[f'loss_{dataloader_idx}'].update(loss, weight=x.size(0))
            for key in self.test_metrics_dict:
                self.test_metrics[f'{key}_{dataloader_idx}'].update(y_hat, y)

    def on_test_epoch_end(self) -> None:
        if self.size_test_dataloaders is None:  # local validation
            test_epoch_loss = self.test_metrics[f'loss'].compute().item()
            self.log(f"test/loss", test_epoch_loss, add_dataloader_idx=False, on_step=False, on_epoch=True)
            self.test_metrics_epoch[f'test/loss'] = test_epoch_loss
            for key in self.test_metrics_dict:
                metric_epoch_loss = self.test_metrics[f'{key}'].compute().item()
                self.log(f"test/{key}", metric_epoch_loss, add_dataloader_idx=False, on_step=False, on_epoch=True)
                self.test_metrics_epoch[f'test/{key}'] = metric_epoch_loss
        else: # validation using multiple val_dataloader
            for i in range(len(self.size_test_dataloaders)):
                test_epoch_loss = self.test_metrics[f'loss_{i}'].compute().item()
                self.log(f"test/loss_{i}", test_epoch_loss, add_dataloader_idx=False, on_step=False, on_epoch=True)
                self.test_metrics_epoch[f'test/loss_{i}'] = test_epoch_loss
                for key in self.test_metrics_dict:
                    metric_epoch_loss = self.test_metrics[f'{key}_{i}'].compute().item()
                    self.log(f"test/{key}_{i}", metric_epoch_loss, add_dataloader_idx=False, on_step=False, on_epoch=True)
                    self.test_metrics_epoch[f'test/{key}_{i}'] = metric_epoch_loss
