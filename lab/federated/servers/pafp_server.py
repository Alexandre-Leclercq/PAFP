import torch
import threading
from typing import Literal, Optional
from typing_extensions import override
from lab.federated.servers import FedAvgServer
from lab.models import DBEModel
from lab.federated.clients import PAFPClient
from lab.models.modules.ema import BiasCorrectedEMA


class PAFPServer(FedAvgServer):

    def __init__(self, global_model, clients: list[PAFPClient],
                 global_statistics_update_mode: Literal['round', 'ema'], alternate_opt: Optional[dict] = None,
                 initialize_with_same_model=True, **kwargs):
        super().__init__(global_model=global_model, clients=clients, **kwargs)
        # init the global model with not hyperparameters as they are only used during learning

        if global_statistics_update_mode not in ('round', 'ema'):
            raise ValueError(f'Unknown global_statistics_update_mode: {global_statistics_update_mode}')

        self.global_model = global_model
        self.alternate_opt = alternate_opt # a dict containing the keys "name" and "params"
        self.global_statistics_update_mode = global_statistics_update_mode
        self.global_statistics = None
        self.clients = clients
        self.initialize_with_same_model = initialize_with_same_model

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
            round_type = self.determine_round_type()
            self.tensorboard_writer.add_text("round_type", str(round_type), global_step=i)
            round_metrics = self.training_round(round_=i, round_type=round_type)
            val_metrics_global_model = self.evaluate_global_model(mode='validation')
            round_metrics = {**round_metrics, **val_metrics_global_model}
            print(f'round_metrics: {round_metrics}')
            for key in round_metrics:
                log_dict[f'{key}'] = round_metrics[key]
            # assessed using round_metrics if global_model weight should be saved for this round
            self.model_checkpointing(round_metrics, round_=i)
            self.save_last_global_weights()
            for logger in self.global_trainer.loggers:
                logger.log_metrics(log_dict, step=i)
        self.end_training()
        # once training is over compute the test metric
        self.test()
        print('end training')

    def average_global_statistics(self, clients_statistics: list[dict], clients_training_size: list) -> list[dict]:
        """

        :param clients_statistics: a list of dim: client x layer. Contain a statictics dict for each client/layer
        :param clients_training_size:

        :return:
        """
        assert len(clients_statistics) == len(clients_training_size)

        global_statistics = [
            {k: torch.zeros_like(v) for k, v in clients_statistics[0][layer].items()}
            for layer in range(len(clients_statistics[0]))
        ]
        for idx_client in range(len(clients_statistics)):
            for idx_layer in range(len(clients_statistics[idx_client])):
                for key in global_statistics[0]:
                    global_statistics[idx_layer][key] += clients_training_size[idx_client]/sum(clients_training_size) * clients_statistics[idx_client][idx_layer][key]
        return global_statistics

    def copy_global_statistics(self) -> None | list[dict]:
        if self.global_statistics is None:
            return None
        return [{k: v.clone().detach() for k, v in layer.items()} for layer in self.global_statistics]

    def set_global_statistics(self, new_global_statistics: list[dict]):
        if self.global_statistics_update_mode == 'round':
            self.global_statistics = new_global_statistics
        elif self.global_statistics_update_mode == 'ema':
            if self.global_statistics is None: # initialize EMA with zeros tensor before first update
                self.global_statistics = [{k: torch.zeros_like(v)  for k, v in new_global_statistics[layer].items()}
                                          for layer in range(len(new_global_statistics))]
            for layer in range(len(new_global_statistics)):
                assert new_global_statistics[layer].keys() == self.global_statistics[layer].keys()
                momentum = 0.8
                for key in new_global_statistics[layer]: # lerp: alpha * old_v + (alpha - 1) * new_value
                    self.global_statistics[layer][key].lerp_(new_global_statistics[layer][key], momentum)

    def log_global_statistic(self, old_global_statistics: list[dict], new_global_statistics: list[dict], round_: int) -> None:
        if old_global_statistics is None:
            return
        assert len(old_global_statistics) == len(new_global_statistics)
        variation_global_statistic = {}
        for layer in range(len(new_global_statistics)):
            assert old_global_statistics[layer].keys() == new_global_statistics[layer].keys()
            variation_global_statistic = {
                **variation_global_statistic,
                **{f'global_statistics_variation/{layer}_{key}': torch.dist(
                    old_global_statistics[layer][key].float(),
                    new_global_statistics[layer][key].float(), p=2
                    ) for key in new_global_statistics[layer].keys()
                }
            }

        for logger in self.global_trainer.loggers:
            logger.log_metrics(variation_global_statistic, step=round_)

    def determine_round_type(self) -> Optional[Literal['opt_global','opt_local']]:
        """
        function that determine which type of round the next round must be.
        :return: Literal['opt_global','opt_local']
        """
        if self.alternate_opt is None:
            return None
        if self.alternate_opt['name'] == 'switch_every_n_rounds':
            if self.current_round > self.alternate_opt['params']['start_after_n_rounds']:
                if self.current_round % self.alternate_opt['params']['switch_n_rounds'] == 0:
                    return 'opt_local'
            return 'opt_global'


    @override
    def client_local_round(self, client, round_: int, round_type, global_weight, local_parameters,
                           clients_statistics, clients_training_size, clients_validation_size,
                           clients_training_metrics, clients_validation_metrics):
        """
        Define the process perform per client. Isolate this process in a dedicated function is usefull for parallelisation.

        :param round_:
        :param client:
        :param global_weight:
        :param local_parameters:
        :param clients_statistics:
        :param clients_training_size:
        :param clients_validation_size:
        :param clients_training_metrics:
        :param clients_validation_metrics:
        :return:
        """
        client_weight, client_statistics, training_data, validation_data = client.local_training(round_, round_type,
                                                                                                 global_weight, self.global_statistics)
        local_parameters[client.client_id] = client_weight
        clients_statistics[client.client_id] = client_statistics
        clients_training_size[client.client_id] = training_data[0]
        clients_training_metrics[client.client_id] = training_data[1]
        clients_validation_size[client.client_id] = validation_data[0]
        clients_validation_metrics[client.client_id] = validation_data[1]

    @override
    def training_round(self, round_, round_type) -> dict:
        """
        perform one training round.
        - send global weight to clients
        - perform local training on clients
        - receive clients updates
        - aggregate clients weight to global weight
        - compute global metrics

        :return: a dict of global metrics
        """
        sampled_clients = self.sample_clients()

        local_parameters = [None] * len(sampled_clients)
        clients_statistics = [None] * len(sampled_clients)
        clients_training_size = [None] * len(sampled_clients)
        clients_validation_size = [None] * len(sampled_clients)
        clients_training_metrics = [None] * len(sampled_clients)
        clients_validation_metrics = [None] * len(sampled_clients)
        global_weight = self.global_model.get_parameters()

        if self.parallelism:  # parallelization case
            threads = []
            for client in sampled_clients:
                t = threading.Thread(target=self.client_local_round, args=(
                    client,
                    round_,
                    round_type,
                    global_weight,
                    local_parameters,
                    clients_statistics,
                    clients_training_size,
                    clients_validation_size,
                    clients_training_metrics,
                    clients_validation_metrics
                    )
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:  # iterative case
            for client in sampled_clients:
                self.client_local_round(client, round_, round_type, global_weight, local_parameters,
                                        clients_statistics, clients_training_size, clients_validation_size,
                                        clients_training_metrics, clients_validation_metrics)

        new_global_weight = self.aggregate_weight(local_parameters, clients_training_size)
        new_global_statistics = self.average_global_statistics(clients_statistics, clients_training_size)
        old_global_statistics = self.copy_global_statistics()
        self.set_global_statistics(new_global_statistics)
        self.log_global_statistic(old_global_statistics=old_global_statistics, new_global_statistics=self.global_statistics, round_=round_)
        self.log_server_weight(global_weight, new_global_weight, local_parameters, round_)
        self.global_model.set_parameters(new_global_weight, round_)

        training_metrics = self.aggregate_metrics_client_set(clients_training_metrics, clients_training_size)
        validation_metrics = self.aggregate_metrics_client_set(clients_validation_metrics, clients_validation_size)

        return {**training_metrics, **validation_metrics}
