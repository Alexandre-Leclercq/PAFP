from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from lab.federated.clients import Client
from lightning.pytorch.utilities.rank_zero import rank_zero_warn
from lab.utils.logging import server_log_pretty_dataset_distribution, server_log_param_cosine_similarity
from pytorch_lightning.utilities import grad_norm
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import Trainer
import os
from pathlib import Path
import torch
from hydra.utils import instantiate
from typing import Literal, Mapping, Optional
from random import sample
from lab.data.datamodules import BaseDataModule


def unfold_metric_dict(target_dict: dict, source_dict: dict, prefix: str):
    """
    parse each metric contain in the source_dict to add with the proper naming in the target_dict.
    E.g. used to unfold test_set metrics from clients and add them to a global test metric dict
    :param target_dict:
    :param source_dict:
    :param prefix:
    :return:
    """
    for metric_name, metric_value in source_dict.items():
        # assuming metric_name is following the set/metric naming convention
        metric = metric_name.split('/')[-1]
        target_dict[f'{prefix}/{metric}'] = metric_value
    return target_dict


class BasicFederatedServer(ABC):

    def __init__(self,
                 clients: list[Client],
                 global_model,
                 global_trainer_cfg: Trainer,
                 parallelism: bool,
                 robust_evaluation: bool,
                 n_round: int,
                 log_dir: str,
                 save_top_k: int,
                 sampling_proportion: float,
                 ckpt_path: Optional[str] = None,
                 **kwargs
                 ):
        """

        :param clients:
        :param global_model:
        :param global_trainer_cfg:
        :param parallelism:
        :param robust_evaluation: boolean that indicates whether to use robust evaluation (client evaluation using each client evaluation set)
        :param n_round:
        :param log_dir:
        :param monitor:
        :param save_top_k:
        :param sampling_proportion:
        :param kwargs:
        """
        super().__init__()
        self.clients = clients
        self.global_model = global_model
        self.current_round = 0
        self.sampling_proportion: float = sampling_proportion  # C

        #############################
        ##  Configure the trainer  ##
        #############################
        if 'tensorboard' in global_trainer_cfg.logger:
            global_trainer_cfg.logger['tensorboard'].save_dir = os.path.join(global_trainer_cfg.logger['tensorboard'].save_dir, 'server')
        if 'csv_logger' in global_trainer_cfg.logger:
            global_trainer_cfg.logger['csv_logger'].save_dir = os.path.join(global_trainer_cfg.logger['csv_logger'].save_dir, 'server')
        tensorboard_index_logger = list(global_trainer_cfg.logger.keys()).index('tensorboard')
        global_trainer_cfg.logger = [logger for logger in global_trainer_cfg.logger.values()]
        self.global_trainer = instantiate(global_trainer_cfg)

        self.parallelism = parallelism
        self.robust_evaluation = robust_evaluation
        self.n_round = n_round
        self.log_dir = log_dir
        self.global_weights_path = os.path.join(log_dir, 'server', 'checkpoints')

        # use for manual checkpointing
        # were used previously for manual checkpoiting of model
        # currently, not used
        self.save_top_k: int = save_top_k
        self.top_metrics = []

        self.tensorboard_writer =self.global_trainer.loggers[tensorboard_index_logger].experiment

        server_log_pretty_dataset_distribution(writer=self.tensorboard_writer, clients=self.clients)

        if ckpt_path is not None:
            self.resume(ckpt_path)

    def get_all_dataloader(self, data_splits: Literal['validation', 'test']) -> list[DataLoader]:
        if data_splits == 'validation':
            return [client.datamodule.val_dataloader() for client in self.clients]
        elif data_splits == 'test':
            return [client.datamodule.test_dataloader() for client in self.clients]
        else:
            raise NotImplementedError(f'data_splits')

    def sample_clients(self):
        if self.sampling_proportion == 1:
            return self.clients
        n_sample_client = int(len(self.clients) * self.sampling_proportion)

        return sample(self.clients, n_sample_client)

    @abstractmethod
    def aggregate_weight(self, local_weight, local_size):
        r"""
        federated which aim to aggregate received weight
        """
        rank_zero_warn("`aggregate_weight` must be implemented")

    def load_global_model(self, global_model_weight_path):
        global_model_weight = torch.load(global_model_weight_path)
        self.global_model.set_parameters(global_model_weight)

    def save_round_global_weights(self, round_: int) -> str:
        os.makedirs(self.global_weights_path, exist_ok=True)
        weights_path = os.path.join(self.global_weights_path, f"global_model_round_{round_}.pt")
        checkpoint = {
            "state_dict": self.global_model.state_dict(),
            "round": round_,
        }
        print(f"saving global weights to {weights_path}")
        torch.save(checkpoint, weights_path)
        return weights_path

    def save_last_global_weights(self) -> None:
        os.makedirs(self.global_weights_path, exist_ok=True)
        last_path = os.path.join(self.global_weights_path, "last.pt")
        checkpoint = {
            "state_dict": self.global_model.state_dict(),
            "round": self.current_round,
        }
        torch.save(checkpoint, last_path)

    def end_training(self):
        for client in self.clients:
            client.end_training()

    def resume(self, ckpt_path: str) -> None:
        """
        Restore the global model weights and the round counter from a checkpoint.
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.global_model.set_parameters(checkpoint["state_dict"], checkpoint["round"])
        # Resume from the round *after* the one that was saved
        self.current_round = checkpoint["round"] + 1
        print(f"Resumed global model from {ckpt_path} — starting at round {self.current_round}")

    def fit(self):
        # Init the client
        for client in self.clients:
            if self.robust_evaluation:
                client.init_fit(validation_dataloaders=self.get_all_dataloader('validation'))
            else: # local evaluation (real-life scenario fl)
                client.init_fit()

        for i in range(self.current_round, self.n_round):
            self.current_round = i
            log_dict = {}
            print(f'round {i}/{self.n_round - 1}')
            # return the local metric of each client computed on local train and val set.
            round_metrics = self.training_round(round_=i)
            print('round metrics received')
            val_metrics_global_model = self.evaluate_global_model(mode='validation')
            round_metrics = {**round_metrics, **val_metrics_global_model}
            print(f'round_metrics: {round_metrics}')
            for key in round_metrics:
                log_dict[f'{key}'] = round_metrics[key]
            # assessed using round_metrics if global_model weight should be saved for this round
            self.model_checkpointing(round_metrics, round_=i)
            self.save_last_global_weights()
            print('checkpoint saved')
            for logger in self.global_trainer.loggers:
                logger.log_metrics(log_dict, step=i)
        print('initiating end of training')
        self.end_training()
        print('end training')

    def test(self, mode='best'):
        """
        Either call at the end of training (No args)
        Or call by a script and thus with a mode and will thus reload client checkpoint.


        Initial intended behaviour: do local testing on local dataset using either best / last checkpoint of each client.
        Note: in case of use of the use of best client checkpoint, the global model produce from wouldn't be naturally
        produce by the system as it's very unlikely that every client have their best epoch at the same moment.
        Thus, a global model produce by such local model could face a very heterogeneous behaviour as the local model
        shouldn't have experience the same number of rounds.
        """
        test_metrics = {}
        clients_test_epoch_metrics = []
        clients_test_size = []
        for client in self.clients:
            # if self.robust_evaluation:
            #     clients_test_epoch_metrics.append(client.test(self.get_all_dataloader(data_splits='test')))
            # else:
            #     clients_test_epoch_metrics.append(client.test())
            # client.fix_checkpoint_paths()
            clients_test_epoch_metrics.append(client.test(ckpt_path=mode))
            test_metrics = unfold_metric_dict(test_metrics, clients_test_epoch_metrics[-1], f'client_{client.client_id}')
            clients_test_size.append(len(client.datamodule.test_dataloader().dataset))
        client_weighted_avg = self.aggregate_metrics_client_set(clients_test_epoch_metrics, clients_test_size, mode='weighted')
        test_metrics = unfold_metric_dict(test_metrics, client_weighted_avg, 'weighted_avg')
        client_unweighted_avg = self.aggregate_metrics_client_set(clients_test_epoch_metrics, clients_test_size, mode='unweighted')
        test_metrics = unfold_metric_dict(test_metrics, client_unweighted_avg, 'unweighted_avg')

        global_model_metrics = self.evaluate_global_model(mode='test')
        print(f'global_model_metrics: {global_model_metrics}')
        test_metrics = {**test_metrics, **global_model_metrics}
        log_dict = {}
        for key in test_metrics:
            log_dict[key] = test_metrics[key]
        for logger in self.global_trainer.loggers:
            logger.log_metrics(log_dict, step=0)
        return test_metrics
        # metrics = {}
        # clients_data_size = []
        # clients_metrics = []
        #
        # for client in self.clients:
        #     checkpoint_path = os.path.join(model_dir, client.client_name, 'checkpoints')
        #     last_ckpt = 'last.ckpt'
        #     checkpoint = os.listdir(checkpoint_path)
        #     checkpoint.remove(last_ckpt)
        #     best_ckpt = checkpoint[0]
        #     if mode == 'best':
        #         local_data_size, metric_local = client.test(os.path.join(checkpoint_path, best_ckpt))
        #     elif mode == 'last':
        #         local_data_size, metric_local = client.test(os.path.join(checkpoint_path, last_ckpt))
        #     else:
        #         raise ValueError(f'mode {mode} is not supported')
        #
        #     metrics[client.client_name] = metric_local
        #
        #     clients_data_size.append(local_data_size)
        #     clients_metrics.append(metric_local)
        # round_metrics = self.aggregate_metrics_client_set(clients_metrics, clients_data_size)
        # metrics['global'] = round_metrics
        # for key in round_metrics:
        #     self.writer.add_scalar(f"{key}_{mode}", round_metrics[key])
        # self.writer.flush()
        # return metrics

    def model_checkpointing(self, round_metrics: dict, round_: int):
        """
        function which test if the metric of the current round is better than the top-k metric of global model.
        If yes save it and replace the worst metric of the top by it.

        Storing global model can be usefull to restart training from a specific checkpoint.

        :param round_metrics:
        :param round:
        """
        if self.global_model is None:
            # Don't execute this function if the global_model is None
            # (It happens in the isolated scenario)
            return

        monitor = self.global_model.monitor

        print(f'server round_metrics: {round_metrics}')
        print(f'server is monitoring {monitor}')
        assert monitor in round_metrics

        # lambda which test if a value worse than the one known as the worst one
        if self.global_model.monitor_objective == 'min':
            is_worst_value = lambda tested_value, worst_value: tested_value > worst_value
        elif self.global_model.monitor_objective == 'max':
            is_worst_value = lambda tested_value, worst_value: tested_value < worst_value
        else:
            raise ValueError(f'Invalid monitor objective: {self.global_model.monitor_objective}')

        round_metric_value = round_metrics[monitor]

        if len(self.top_metrics) < self.save_top_k:
            self.top_metrics.append({
                "metric": round_metric_value,
                "round": round_,
                "checkpoint_path": self.save_round_global_weights(round_),
            })
            return

        worst_index = 0
        for i in range(len(self.top_metrics)):  # loop to find worst indice in
            if is_worst_value(self.top_metrics[i]['metric'], self.top_metrics[worst_index]['metric']):
                worst_index = i
        # if true it means the round metric is better than the lower in top metric. Thus, replace it by the round metric
        if is_worst_value(self.top_metrics[worst_index]['metric'], round_metric_value):
            print(f'new best {monitor} : {round_metric_value}')
            # delete the weight of the round that is not in the top metric anymore
            os.remove(self.top_metrics[worst_index]['checkpoint_path'])
            # replace the current round metric by one who's leavin the top-k value
            self.top_metrics[worst_index] = {
                "metric": round_metric_value,
                "round": round_,
                "checkpoint_path": self.save_round_global_weights(round_),
            }

    def aggregate_metrics_client_set(self, clients_metrics: list[Mapping[str, float]],
                                     clients_dataset_size: Optional[list[int]] = None,
                                     mode: Literal['weighted', 'unweighted'] = 'weighted'):
        """
        use to aggregate the produced metric by each client local dataset to form a global dataset metrics
        mode:
        - weighted: performed a weighted average
        - unweighted: performed an unweighted average
        """
        if mode == 'weighted':
            assert clients_dataset_size is not None
            assert len(clients_metrics) == len(clients_dataset_size)
        metric_keys = clients_metrics[0].keys()
        aggregated_metrics = {}
        for metric_key in metric_keys:
            if mode == 'weighted':
                aggregated_metrics[f'{metric_key}'] = 0
                for i, client_metrics in enumerate(clients_metrics):
                    aggregated_metrics[f'{metric_key}'] += clients_dataset_size[i] * client_metrics[metric_key]
                aggregated_metrics[f'{metric_key}'] /= sum(clients_dataset_size)
            elif mode == 'unweighted':
                aggregated_metrics[f'{metric_key}_{mode}'] = 0
                for i, client_metrics in enumerate(clients_metrics):
                    aggregated_metrics[f'{metric_key}_{mode}'] += client_metrics[metric_key]
                aggregated_metrics[f'{metric_key}_{mode}'] /= len(clients_metrics)
            else:
                raise ValueError(f'mode {mode} is not supported')
        return aggregated_metrics

    def evaluate_global_model(self, mode: Literal['validation', 'test']):
        """
        use the test set of each client to evaluate the global model.
        """
        if self.global_model is None:
            # Don't execute this function if the global_model is None
            # (It happens in the isolated scenario)
            return {}

        list_dataloaders = self.get_all_dataloader(data_splits=mode)
        size_dataloaders = [len(dataloader.dataset) for dataloader in list_dataloaders]
        if mode == 'validation':
            # set the global step of the trainer as round so that the metrics of the global model are logged per round
            self.global_trainer.validate(self.global_model, list_dataloaders)
            results = self.global_model.val_metrics_epoch
        elif mode == 'test':
            best_ckpt_path = next(p for p in Path(self.global_weights_path).iterdir() if p.suffix in ('.ckpt', '.pt'))
            checkpoint = torch.load(best_ckpt_path, map_location=lambda storage, loc: storage)
            self.global_model.load_state_dict(checkpoint['state_dict'], strict=False)
            results: list[Mapping[str, float]] = self.global_trainer.test(self.global_model, dataloaders=list_dataloaders)[0]
            print(f'results: {results}')
        else:
            raise ValueError(f'mode {mode} is not supported')

        processed_results = {}
        # retrieve the unique key of each metrics
        metric_keys = set([''.join(key.split('_')[:-1]) for key in results.keys()])

        # aggregate the metric of the global model
        for metric in metric_keys:
            processed_results[f'{metric}_global_model'] = 0
            for i, size_dataloader in enumerate(size_dataloaders):
                processed_results[f'{metric}_global_model'] += size_dataloader * results[f'{metric}_{i}']
            processed_results[f'{metric}_global_model'] /= sum(size_dataloaders)
        return processed_results

    def log_server_weight(self, global_weight, new_global_weight, local_parameters: list, round_: int):
        """

        :param global_weight: weight of global model before aggregation
        :param new_global_weight: weight of global model after aggregation
        :param local_parameters: list of each client shared parameters. We assume the list follow the client index order
        :return:
        """
        n_clients = len(self.clients)
        if global_weight.keys() != new_global_weight.keys():
            for k in global_weight.keys():
                if k not in new_global_weight.keys():
                    print(f'global_weight: {k} not in new_global_weight')
            for k in new_global_weight.keys():
                if k not in global_weight.keys():
                    print(f'new_global_weight: {k} not in global_weight')
            raise KeyError('global_weight and new_global_weight keys do not match')

        variation_global_model = {f'global_model_variation/{key}': torch.dist(global_weight[key].float(), new_global_weight[key].float(), p=2) for key in global_weight.keys()}

        for logger in self.global_trainer.loggers:
            logger.log_metrics(variation_global_model, step=round_)

        assert len(local_parameters) == n_clients

        if round_ % 10 == 0: # compute it every 10 round to reduce computation cost
            server_log_param_cosine_similarity(self.tensorboard_writer, n_clients=n_clients,
                                           clients_parameters=local_parameters, new_global_weight=new_global_weight,
                                           step=round_)

        for i in range(n_clients):
            client = self.clients[i]
            assert i == client.client_id
            client_consensus = {
                f'global_model_consensus/{key}':
                torch.dist(local_parameters[i][key].float(), new_global_weight[key].float(), p=2)
                for key in local_parameters[i].keys()
            }

            for logger in client.trainer.loggers:
                logger.log_metrics(client_consensus, step=round_)

    @abstractmethod
    def training_round(self, round_: int):
        r"""
        federated which perform one round of communication
        1. sample_cleint
        2. send global model to client
        3. perform training on client
        4. receive clients weight
        5. aggregate clients weight
        """
        rank_zero_warn("`training_round` must be implemented")
