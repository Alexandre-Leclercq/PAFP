import logging
import os
from collections import OrderedDict
import torch
from scipy.cluster.hierarchy import weighted
from torch import nn
import timm
from lab.models import BaseFedModel
from lab.utils import instantiate_from_class_path
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
Based on : https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/model.py
Baseline model implementation for Fed-ISIC2019
"""
class EfficientNet(BaseFedModel):
    def __init__(self,
                 learning_rate: float,
                 optimizer: str,
                 weight_decay: float,
                 dropout_rate: float,
                 weighted_loss: bool,
                 timm_dir: str,
                 monitor: str,
                 monitor_objective: str,
                 loss: type,
                 in_channels: int,
                 img_size: int,
                 n_class: int,
                 train_metrics: dict,
                 validation_metrics: dict,
                 test_metrics: dict,
                 gradient_clip_val: float = 0.0,
                 gradient_logging: bool = False,
                 pretrained: bool = True,
                 freeze_encoder: bool = False) -> None:
        super(EfficientNet, self).__init__(loss, train_metrics, validation_metrics, test_metrics,
                                           monitor, monitor_objective,
                                           gradient_logging, weighted_loss)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.dropout_rate = dropout_rate
        self.weighted_loss = weighted_loss
        self.img_size = img_size
        self.pretrained = pretrained
        self.model_name = 'efficientnet_b0.ra_in1k'
        self.freeze_encoder = freeze_encoder

        self.encoder = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=os.path.join(timm_dir, f'{self.model_name}.safetensors') if pretrained else None),
            in_chans=in_channels,
            num_classes=0,  # No classifier
            global_pool='', # No pooling inside the encoder
        )

        if freeze_encoder:
            assert pretrained # we can't freeze the encoder if we don't used the pretrained version
            # freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()  # set the encoder in inference mode

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.latent_space = 1280

        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.encoder.num_features, n_class)
        )

    def get_hyperparameters(self) -> dict:
        """
        OBSOLETE
        return a dictionary of hyperparameters.
        this function is used when the current model is wrapped in another model
        :return:
        """
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'dropout_rate': self.dropout_rate,
        }

    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """
        get model parameter and put them on cpu before giving them.
        We make sure to detach the tensor
        :return:
        """
        model_params = OrderedDict(
            (k, v.detach().cpu().clone())
            for k, v in self.state_dict().items()  # Use named_parameters, not state_dict
        )
        return model_params

    def set_parameters(self, parameters, current_round: int):
        """
        use to update local weight from the one received
        """
        self.current_round = current_round
        self.load_state_dict(parameters)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.gap(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.head(z)
        return y

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encode(x)
        y_hat = self.decode(z)
        loss = self.loss(y_hat, y)
        self.log_images(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for key in self.train_metrics:
            metric = self.train_metrics[key](y_hat, y)
            self.log(f"train/{key}", metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'latents': z, 'targets': y}

    def configure_optimizers(self):
        optimizer = instantiate_from_class_path(self.optimizer,
                                                {
                                                    'params':self.parameters(),
                                                    'lr': self.learning_rate,
                                                    'weight_decay': self.weight_decay
                                                })
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5
        # )
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val/loss',
            # }
        }