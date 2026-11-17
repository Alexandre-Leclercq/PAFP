from hydra.utils import instantiate
from pathlib import Path
from lab.data.datamodules import BaseDataModule
from torch.utils.data import DataLoader
from lab.models import BaseFedModel
from lab.utils.logging import client_log_pretty_dataset_distribution, model_log_pretty_class_weight
from typing import Optional, Literal, Mapping
from collections import OrderedDict
import logging
import torch
import os


class Client:

    def __init__(self,
                 client_id: int,
                 model: BaseFedModel,
                 datamodule: BaseDataModule,
                 trainer_cfg,
                 robust_evaluation: bool,
                 mode: Literal['train', 'resume', 'test'],
                 device_id: Optional[int] = None,
                 steps_mode: Literal['iter', 'epoch'] = 'epoch',
                 max_steps: Optional[int] = None,
                 ckpt_path: Optional[str] = None):
        print(f'initializing client {client_id}')
        # will evolve regarding how we implement the data partitioning
        # for now we just pass a datamodule
        self.client_id = client_id
        self.client_name = f'client_{client_id}'
        self.datamodule = datamodule
        self.datamodule.prepare_data()
        self.mode = mode
        self.robust_evaluation = robust_evaluation
        self.ckpt_path = ckpt_path
        if mode == 'train' or mode == 'resume':
            self.datamodule.setup(stage="fit")  # setup manually to avoid the trainer do it at every round
        elif mode == 'test':
            self.datamodule.setup(stage="test")
        else:
            raise NotImplementedError

        self.model = model

        # set class_weights for model that expect it (weighted_loss) and datamodule that have can handle it
        class_weights_set = False
        if hasattr(self.model, 'weighted_loss') and hasattr(datamodule, 'class_weights'):
            print('setting class_weights to model')
            class_weights_set = self.model.set_class_weights(datamodule.class_weights)

        self.current_round = None

        #############################
        ##  Configure the trainer  ##
        #############################
        self.max_steps = max_steps # E local steps (either iters or epochs)
        self.steps_mode = steps_mode
        if steps_mode == 'iter':
            trainer_cfg.max_steps = max_steps
        elif steps_mode == 'epoch':
            trainer_cfg.max_epochs = max_steps
        else:
            raise ValueError(f'invalid steps_mode: {steps_mode}')

        if device_id is not None:
            trainer_cfg.devices = [device_id]
        # trainer_cfg.logger.root_dir = os.path.join(trainer_cfg.logger.root_dir, self.client_name)

        if 'tensorboard' in trainer_cfg.logger:
            trainer_cfg.logger['tensorboard'].save_dir = os.path.join(trainer_cfg.logger['tensorboard'].save_dir, self.client_name)
        if 'csv_logger' in trainer_cfg.logger:
            trainer_cfg.logger['csv_logger'].save_dir = os.path.join(trainer_cfg.logger['csv_logger'].save_dir, self.client_name)
        tensorboard_index_logger = list(trainer_cfg.logger.keys()).index('tensorboard')
        trainer_cfg.logger = [logger for logger in trainer_cfg.logger.values()]

        callbacks = trainer_cfg.callbacks
        # multiple dataset are going to be furnished, but we save the best client model
        # based on its local validation set.
        if robust_evaluation:
            callbacks.model_checkpoint.monitor += f'_{client_id}'

        # remove the checkpoint subdir so that we could put it within the self.client_name dir
        callbacks.model_checkpoint.dirpath = os.path.dirname(callbacks.model_checkpoint.dirpath)
        callbacks.model_checkpoint.dirpath = os.path.join(callbacks.model_checkpoint.dirpath, self.client_name,
                                                          "checkpoints")
        print(f'callbacks.model_checkpoint.dirpath: {callbacks.model_checkpoint.dirpath}')

        if 'latent_saver' in callbacks:
            callbacks.latent_saver.prefix_name = f'c{client_id}'

        trainer_cfg.callbacks = list(callbacks.values())
        self.trainer = instantiate(trainer_cfg)

        self.logger = logging.getLogger(__name__)

        self.tensorboard_writer = self.trainer.loggers[tensorboard_index_logger].experiment

        client_log_pretty_dataset_distribution(writer=self.tensorboard_writer, datamodule=self.datamodule)
        if class_weights_set:
            model_log_pretty_class_weight(writer=self.tensorboard_writer, class_weights=datamodule.class_weights)

    """
    Training Methods
    """
    def init_fit(self, validation_dataloaders: Optional[list[DataLoader]] = None):
        # assert self.mode == 'train' or self.mode == 'resume'

        if validation_dataloaders is None: # only use local data
            self.trainer.init_fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        else:
            # initialise model for training and pass it the list of val_dataloaders
            # val_dataloaders are sort in client order on the server side
            self.trainer.init_fit(self.model, train_dataloaders=self.datamodule.train_dataloader(),
                              val_dataloaders=validation_dataloaders, ckpt_path=self.ckpt_path)

    def local_training(self, round_: int, global_weight: OrderedDict[str, torch.Tensor]) -> tuple[
        OrderedDict[str, torch.Tensor],
        tuple[int, dict],
        tuple[int, dict]
    ]:
        """
        perform one round of training locally and return the new weight, local_dataset_size and computed_metrics

        :param round_:
        :param global_weight: global weight receive from server

        :type global_weight: OrderedDict[str, torch.Tensor]
        :return:
            - new local weight learn
            - size of training data
            - ModuleDict of computed metrics

        :rtype tuple[OrderedDict[str, torch.Tensor], int, torch.nn.ModuleDict]
        """
        self.current_round = round_
        self.model.set_parameters(global_weight, round_)
        print(f'beginning training {self.client_name}')
        if self.steps_mode == 'epoch':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} epochs')
            self.trainer.partial_fit_epochs(epochs=self.max_steps)
        elif self.steps_mode == 'iter':
            self.logger.info(f'starting training {self.client_name} for {self.max_steps} iterations')
            self.trainer.partial_fit_iterations(iterations=self.max_steps)

        print(f'end training {self.client_name}')
        train_metrics = self.model.train_metrics_epoch
        validation_metrics = self.processing_validation_metric(self.model.val_metrics_epoch)
        print(f'train_metrics: {train_metrics}')
        print(f'validation_metrics: {validation_metrics}')
        print(f'end process val metrics for {self.client_name}')
        return (self.model.get_parameters(),
                (len(self.datamodule.train_dataloader().dataset), train_metrics),
                (len(self.datamodule.train_dataloader().dataset), validation_metrics))

    def end_training(self):
        self.trainer.complete_fit()

    def processing_validation_metric(self, validation_metrics):
        """
        This function aim to process metric to have the right format before sending them to the server.
        1. aggregate each local metric to forme the metric_global_set
        2. reformat the local metric_i to just metric
        """

        processed_validation_metric = {}
        log_dict = {}
        if self.model.size_validation_dataloaders is None:
            metric_keys = validation_metrics.keys()
            for metric in metric_keys:
                processed_validation_metric[f'{metric}'] = validation_metrics[f'{metric}']
                log_dict[f'{metric}'] = processed_validation_metric[f'{metric}']
        else:
            # retrieve the unique key of each metrics (in case of multi dataloaders)
            metric_keys = set([''.join(key.split('_')[:-1]) for key in validation_metrics.keys()])
            for metric in metric_keys:
                processed_validation_metric[f'{metric}'] = validation_metrics[f'{metric}_{self.client_id}']
                log_dict[f'{metric}'] = processed_validation_metric[f'{metric}']
                global_set_metric = 0
                for i, size_validation_dataloader in enumerate(self.model.size_validation_dataloaders):
                    global_set_metric += size_validation_dataloader * validation_metrics[f'{metric}_{i}']
                global_set_metric /= sum(self.model.size_validation_dataloaders)
                log_dict[f'{metric}_global_set'] = global_set_metric
                processed_validation_metric[f'{metric}_global_set'] = global_set_metric
        for logger in self.trainer.loggers:
            logger.log_metrics(log_dict, step=self.current_round)
        return processed_validation_metric

    # def fix_checkpoint_paths(self) -> None:
    #     """
    #     Replace the stale absolute dirpath stored in ModelCheckpoint callback state
    #     (from the training machine) with the local checkpoints directory.
    #     Called before test() when running on a different machine than training.
    #     """
    #     local_ckpt_dir = Path(self.trainer.checkpoint_callback.dirpath)
    #     ckpt_callback = self.trainer.checkpoint_callback
    #
    #     if ckpt_callback.best_model_path:
    #         filename = Path(ckpt_callback.best_model_path).name
    #         ckpt_callback.best_model_path = str(local_ckpt_dir / filename)
    #
    #     if ckpt_callback.last_model_path:
    #         filename = Path(ckpt_callback.last_model_path).name
    #         ckpt_callback.last_model_path = str(local_ckpt_dir / filename)

    def test(self, test_dataloaders: Optional[list[DataLoader]]=None, ckpt_path="best") -> dict:
        """
        Compute the testing of client either on providen dataloaders or local test set if none if providen
        :param test_dataloaders:
        :param ckpt_path: ckpt_path used by lightning trainer
        :return: test metrics
        """
        if self.mode == 'test':
            if test_dataloaders is not None:
                self.trainer.test(model=self.model, dataloaders=test_dataloaders, ckpt_path=self.ckpt_path)
            else:
                self.trainer.test(model=self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=self.ckpt_path)
        elif self.mode == 'train' or self.mode == 'resume': # testing after training
            if test_dataloaders is not None:
                self.trainer.test(dataloaders=test_dataloaders, ckpt_path=ckpt_path)
            else:
                self.trainer.test(dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)
        else:
            raise NotImplementedError(f'Unknown mode {self.mode}')

        test_metrics = self.model.test_metrics_epoch
        processed_test_metric = {}
        log_dict = {}
        if self.model.size_test_dataloaders is None:
            metric_keys = test_metrics.keys()
            for metric in metric_keys:
                processed_test_metric[f'{metric}'] = test_metrics[f'{metric}']
                log_dict[f'{metric}'] = processed_test_metric[f'{metric}']
        else:
            metric_keys = set([''.join(key.split('_')[:-1]) for key in test_metrics.keys()])
            for metric in metric_keys:
                processed_test_metric[f'{metric}'] = test_metrics[f'{metric}_{self.client_id}']
                log_dict[f'{metric}'] = processed_test_metric[f'{metric}']
                global_set_metric = 0
                for i, size_test_dataloader in enumerate(self.model.size_test_dataloaders):
                    global_set_metric += size_test_dataloader * test_metrics[f'{metric}_{i}']
                global_set_metric /= sum(self.model.size_test_dataloaders)
                log_dict[f'{metric}_global_set'] = global_set_metric
                processed_test_metric[f'{metric}_global_set'] = global_set_metric
        for logger in self.trainer.loggers:
            logger.log_metrics(log_dict, step=self.current_round)
        return processed_test_metric
