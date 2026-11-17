import os
from collections import OrderedDict
import torch
from typing import Optional, Sequence, Literal
from lab.models.modules import PersonalizeAffineConvModule
from torch import nn
import timm
from lab.models import BaseFedModel
from lab.models.modules.custom_layers import HEADAffine
from lab.utils import instantiate_from_class_path
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

def inject_personalized_submodule_in_efficient_net(efficient_net_encoder, image_size, unfrozen_params, submodule_to_inject, module_params: dict):
    """
    Inject submodule at the beginning of unfrozen EfficientNet Blocks
    :param image_size:
    :param efficient_net_encoder:
    :param unfrozen_params:
    :param submodule_to_inject:
    :param module_params:
    :return:
    """
    print(f'inject {submodule_to_inject} in efficient_net_encoder')
    new_blocks = nn.Sequential()
    personalized_layers_name = []
    personalized_layers = []
    dummy_input = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        dummy_input = efficient_net_encoder.conv_stem(dummy_input)
        for i, (name, stage) in enumerate(efficient_net_encoder.blocks.named_children()):
            full_name_stage = f'blocks.{name}'
            if full_name_stage in unfrozen_params:
                print(f'dummy input of {stage[0].__class__.__name__} is {dummy_input.shape}')
                module_params['features_shape'] = tuple(dummy_input.shape[1:])
                module_params['num_groups'] = dummy_input.shape[1]//8 # EfficientNet channels only divisible by 8 (16, 24, 40, 80, 112, 192)
                added_module = submodule_to_inject(**module_params)
                stage = nn.Sequential(
                    added_module,
                    *stage
                )
                personalized_layers_name.append(f'{full_name_stage}.{0}')
                personalized_layers.append(added_module)
            dummy_input = stage(dummy_input)

            new_blocks.add_module(f"{i}", stage)
    efficient_net_encoder.blocks = new_blocks
    return efficient_net_encoder, personalized_layers, personalized_layers_name

"""
PFP (Personalized Aligner Federated Finetuned) EfficientNet
"""
class PAFP_EfficientNet_2Opt_Joint(BaseFedModel):
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
                 lambda_: float,
                 personalized_features_aligner: dict = dict(),
                 gradient_clip_val: float = 0.0,
                 gradient_logging: bool = False,
                 personalized_logits: bool = False,
                 unfrozen_params: Sequence[Optional['str']]= (),
                 local_params_keys: Sequence[Optional['str']]= ()) -> None:
        super(PAFP_EfficientNet_2Opt_Joint, self).__init__(loss, train_metrics, validation_metrics, test_metrics,
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
        self.lambda_ = lambda_

        self.opt_mode: Literal['opt_global', 'opt_local'] = 'opt_global'

        self.encoder = timm.create_model(
            self.model_name,
            pretrained=True,
            pretrained_cfg_overlay=dict(file=os.path.join(timm_dir, f'{self.model_name}.safetensors')),
            in_chans=in_channels,
            num_classes=0,  # No classifier
            global_pool='', # No pooling inside the encoder
        )

        features_aligner_params = dict(
            features_shape=None,
            num_groups=None,
            momentum=0.9,
            statistic_mode='mean-std',
            initialization='id_init'
        )
        assert personalized_features_aligner.keys() - features_aligner_params.keys() == set()
        # override default value by provided params
        for key in personalized_features_aligner:
            features_aligner_params[key] = personalized_features_aligner[key]

        self.encoder, align_modules, personalized_module_keys = inject_personalized_submodule_in_efficient_net(
            self.encoder,
            img_size,
            self.unfrozen_params,
            submodule_to_inject=PersonalizeAffineConvModule,
            module_params=features_aligner_params
        )
        self.align_modules: list[PersonalizeAffineConvModule] = align_modules
        L = len(align_modules)
        beta = 1e-3 # source "Federated Learning with Domain Shift Eraser" 1e-3 equiv to equal way
        self.w_l = torch.exp(beta * torch.arange(L)) / torch.sum(torch.exp(beta * torch.arange(L)))
        self.local_params_keys = (*self.local_params_keys, *personalized_module_keys)

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

        self.automatic_optimization = False

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

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # while lock the BatchNorms/Dropout, etc.
        self.encoder.eval()

        for module_name, module in self.encoder.named_modules():
            if module_name.startswith(self.unfrozen_params):
                module.train()

        for module in self.align_modules:
            module.reset_local_statistics()

    def set_aligner_global_statistics(self, global_statistics: list[dict]):
        for i, module in enumerate(self.align_modules):
            module.set_global_statistics(global_statistics[i])

    def get_aligner_global_statistics(self) -> list[dict]:
        global_statistics = []
        for module in self.align_modules:
            global_statistics.append(module.get_local_statistics())
        return global_statistics

    def training_step(self, batch, batch_idx):
        with torch.autograd.set_detect_anomaly(True):
            base_optimizer, feature_aligner_optimizer = self.optimizers()

            x, y = batch
            z = self.encode(x)
            y_hat = self.decode(z)

            loss_task = self.loss(y_hat, y)
            loss_align = 0
            for i, module in enumerate(self.align_modules):
                assert module.loss is not None
                loss_align += self.w_l[i] * module.loss
            rescal_loss_align = self.lambda_ * loss_align

            base_optimizer.zero_grad()
            feature_aligner_optimizer.zero_grad()

            if self.opt_mode == 'opt_global':
                self.manual_backward(loss_task)
                base_optimizer.step()
            elif self.opt_mode == 'opt_local':
                assert self.current_round > 0
                self.manual_backward((loss_task + rescal_loss_align))
                feature_aligner_optimizer.step()

            loss = loss_task + rescal_loss_align
            self.log_images(batch)
            self.log("train/loss_align", loss_align, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            for key in self.train_metrics:
                metric = self.train_metrics[key](y_hat, y)
                self.log(f"train/{key}", metric, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            return {'loss': loss, 'latents': z, 'targets': y}

    def configure_optimizers(self):
        special_params_ids = set()
        for layer in self.align_modules:
            for param in layer.parameters():
                special_params_ids.add(id(param))

        base_params = []
        feature_aligner_params = []

        for p in self.parameters():
            if id(p) in special_params_ids:
                feature_aligner_params.append(p)
            else:
                base_params.append(p)

        base_optimizer = instantiate_from_class_path(self.optimizer,
                                                {
                                                    'params': base_params,
                                                    'lr': self.learning_rate,
                                                    'weight_decay': self.weight_decay
                                                })

        optimizer_personalized_features_aligner = instantiate_from_class_path(self.optimizer,
                                                {
                                                    'params': feature_aligner_params,
                                                    'lr': self.learning_rate,
                                                    'weight_decay': self.weight_decay
                                                })
        return base_optimizer, optimizer_personalized_features_aligner
