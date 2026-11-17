import logging
import os
from collections import OrderedDict
import torch
from typing import Optional, Sequence
from scipy.cluster.hierarchy import weighted
from torch import nn
import timm
from lab.models import BaseFedModel
from lab.models.modules.custom_layers import HEADAffine
from lab.utils import instantiate_from_class_path
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
PFP (Personalized Federated Finetuned) EfficientNet
"""
class PFP_EfficientNet(BaseFedModel):
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
                 personalized_logits: bool = False,
                 unfrozen_params: Sequence[Optional['str']]= (),
                 local_params_keys: Sequence[Optional['str']]= ()) -> None:
        super(PFP_EfficientNet, self).__init__(loss, train_metrics, validation_metrics, test_metrics,
                                               monitor, monitor_objective,
                                               gradient_logging, weighted_loss, local_params_keys)

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.dropout_rate = dropout_rate
        self.weighted_loss = weighted_loss
        self.img_size = img_size
        self.unfrozen_params = tuple(unfrozen_params)
        self.model_name = 'efficientnet_b0.ra_in1k'

        self.encoder = timm.create_model(
            self.model_name,
            pretrained=True,
            pretrained_cfg_overlay=dict(file=os.path.join(timm_dir, f'{self.model_name}.safetensors')),
            in_chans=in_channels,
            num_classes=0,  # No classifier
            global_pool='', # No pooling inside the encoder
        )

        # Freeze every params of the encoder except the unfrozen layers
        for param_name, param in self.encoder.named_parameters():
            if param_name.startswith(self.unfrozen_params):
                param.requires_grad = True
                print(f'{param_name} is unfrozen')
            else:
                param.requires_grad = False
        print()


        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.latent_space = 1280

        self.head = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.encoder.num_features, n_class)
        )
        self.local_params_keys = (*self.local_params_keys, 'personalized_rescal_logits')
        if personalized_logits:
            self.personalized_rescal_logits = HEADAffine(num_classes=n_class)
        else:
            self.personalized_rescal_logits = nn.Identity()

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # while lock the BatchNorms/Dropout, etc.
        self.encoder.eval()

        for module_name, module in self.encoder.named_modules():
            if module_name.startswith(self.unfrozen_params):
                module.train()

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
        We make sure to detach the tensor.
        Keep local the params mention in local_params_keys.
        :return:
        """
        # list the params to drop (not communicate)
        encoder_param_name = tuple(f'encoder.{param}' for param in self.unfrozen_params)
        local_encoder_param_name = tuple(f'encoder.{param}' for param in self.local_params_keys)
        model_params = OrderedDict(
            (k, v.detach().cpu().clone())
            for k, v in self.state_dict().items() # Use named_parameters, not state_dict
            if not k.startswith(local_encoder_param_name) # exclude parameters having the mentioned prefix name
            and k.startswith(encoder_param_name)
        )
        return model_params

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.gap(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.head(z)
        y = self.personalized_rescal_logits(y)
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