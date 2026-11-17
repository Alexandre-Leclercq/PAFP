from lab.models.modules import BaseFedModule
from lab.models.modules.custom_layers import id_init_conv
from typing import Optional, Literal
import torch
from torch import nn

class PersonalizeAffineConvModule(BaseFedModule):
    def __init__(self, features_shape: tuple,
                 num_groups:int,
                 momentum:float,
                 statistic_mode: Literal['mean-std', 'mean-var'],
                 initialization: Optional[Literal['id_init']]=None,
                 *args,
                 **kwargs):
        super(PersonalizeAffineConvModule, self).__init__()
        assert len(features_shape) == 3 # (C, H, W)
        assert num_groups < features_shape[0]

        self.features_shape = features_shape
        self.momentum = momentum
        self.statistics_mode = statistic_mode
        self.loss = None

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=features_shape[0])
        self.conv = nn.Conv2d(in_channels=features_shape[0], out_channels=features_shape[0], kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU()

        if initialization == 'id_init':
            self.conv.apply(id_init_conv)

        if statistic_mode == 'mean-std':
            self.register_buffer('local_mean', torch.zeros(features_shape, device=self.device))
            self.register_buffer('local_std', torch.zeros(features_shape, device=self.device))
            self.register_buffer('global_mean', torch.full(features_shape, float('nan')))
            self.register_buffer('global_std', torch.full(features_shape, float('nan')))
            self.statistics = [
                {'name': 'mean', 'local_attr': 'local_mean', 'global_attr': 'global_mean',
                 'function': torch.mean, 'function_arg': {'dim':0}},
                {'name': 'std', 'local_attr': 'local_std', 'global_attr': 'global_std',
                 'function': torch.std, 'function_arg': {'dim':0}},
            ]
        elif statistic_mode == 'mean-var':
            self.register_buffer('local_mean', torch.zeros(features_shape, device=self.device))
            self.register_buffer('local_var', torch.zeros(features_shape, device=self.device))
            self.register_buffer('global_mean', torch.full(features_shape, float('nan')))
            self.register_buffer('global_var', torch.full(features_shape, float('nan')))
            self.statistics = [
                {'name': 'mean', 'local_attr': 'local_mean', 'global_attr': 'global_mean',
                 'function': torch.mean, 'function_arg': {'dim':0}},
                {'name': 'std', 'local_attr': 'local_var', 'global_attr': 'global_var',
                 'function': torch.var, 'function_arg': {'dim':0}},
            ]
        else:
            raise NotImplementedError(f'{statistic_mode} is not implemented')

    def reset_local_statistics(self) -> None:
        for statistic in self.statistics:
            getattr(self, statistic['local_attr']).fill_(0)

    def forward(self, x) -> torch.Tensor:
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)

        if self.training:
            self.loss = 0.0
            for statistic in self.statistics:
                new_local_stat = self.compute_new_statistic_mean(statistic, x)
                self.loss += self.loss_reg(statistic, new_local_stat)
                self.set_new_local_statistic(statistic, new_local_stat)
        return x

    def set_new_local_statistic(self, statistic: dict, new_local_stat_value: torch.Tensor) -> None:
            getattr(self, statistic['local_attr']).copy_(new_local_stat_value.detach())

    def compute_new_statistic_mean(self, statistic: dict, module_output: torch.Tensor) -> torch.Tensor:
        current_stat = getattr(self, statistic['local_attr'])
        running_stat = statistic['function'](module_output, **statistic['function_arg'])
        # (1 - weight) * input + weight * end
        return torch.lerp(input=current_stat, end=running_stat.to(current_stat.dtype), weight=self.momentum)

    def loss_reg(self, statistic: dict, new_local_stat) -> torch.Tensor:
        global_stat = getattr(self, statistic['global_attr'])
        if torch.isnan(global_stat).all():
            return torch.tensor(.0, device=self.device)
        else:
            return torch.mean((new_local_stat - global_stat)**2)

    def set_global_statistics(self, new_global_statistics: dict) -> None:
        for statistic in self.statistics:
            assert statistic['name'] in new_global_statistics
            new_val = new_global_statistics[statistic['name']].clone().detach()
            getattr(self, statistic['global_attr']).copy_(new_val)

    def get_local_statistics(self) -> dict:
        local_statistics = {}
        for statistic in self.statistics:
            local_stat = getattr(self, statistic['local_attr'])
            local_statistics[statistic['name']] = local_stat.clone().detach()
        return local_statistics
