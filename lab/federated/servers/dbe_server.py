import torch
import threading
from typing_extensions import override
from lab.federated.servers import FedAvgServer
from lab.models import DBEModel
from lab.federated.clients import DBEClient


class DBEServer(FedAvgServer):
    """
    An extension of FedAvg that implements the DBE method.

    Adding:
    - compute tbe global mean (z^g mean in DBE paper) between each round
    - communicate the global_mean at the beginning of each round (the same way we already did w/ global weight)
    """

    def __init__(self, global_model, clients: list[DBEClient], initialize_with_same_model=True, **kwargs):
        # global_model = DBEModel(global_model, 0, 0, 0)
        super().__init__(global_model=global_model, clients=clients, **kwargs)
        # init the global model with not hyperparameters as they are only used during learning
        self.global_model = global_model
        self.global_mean = None
        self.clients = clients
        self.initialize_with_same_model = initialize_with_same_model

    def fit(self):
        print('Beginning of initialisation round')
        self.compute_global_mean()
        # Assign the global mean to all clients
        print('assigning global mean to every client')
        for client in self.clients:
            client.set_global_mean(self.global_mean.detach().clone())

        # evaluate for round 0

        print('beginning of training')
        for i in range(self.current_round, self.n_round):
            self.current_round = i
            log_dict = {}
            print(f'round {i}/{self.n_round - 1}')
            # return the local metric of each client computed on local train and val set.
            round_metrics = self.training_round(round_=i)
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

    def compute_client_global_mean(self, client, global_weight, clients_running_mean, clients_training_size):
        """
        compute the client specific mean (one epoch perform without DBE)

        :param client:
        :param global_weight:
        :param clients_running_mean:
        :param clients_training_size:
        :return:
        """
        client_running_mean, client_training_size = client.compute_specific_mean(global_weight)
        clients_running_mean.append(client_running_mean)
        clients_training_size.append(client_training_size)

    def compute_global_mean(self):
        """
        compute the global mean by performing one training epoch on sampled client
        :return:
        """
        for client in self.clients:
            if self.robust_evaluation:
                client.init_fit(validation_dataloaders=self.get_all_dataloader('validation'))
            else:
                client.init_fit()

        sampled_clients = self.sample_clients()

        clients_running_mean = []
        clients_training_size = []
        global_weight = self.global_model.get_parameters() if self.initialize_with_same_model else None

        if self.parallelism:
            threads = []
            for client in sampled_clients:
                t = threading.Thread(target=self.compute_client_global_mean, args=(
                    client,
                    global_weight,
                    clients_running_mean,
                    clients_training_size
                    )
                )
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        else:  # iterative case
            for client in sampled_clients:
                self.compute_client_global_mean(client, global_weight, clients_running_mean, clients_training_size)
        self.global_mean = self.average_global_mean(clients_running_mean, clients_training_size)

    def average_global_mean(self, clients_running_mean: list[torch.Tensor], clients_training_size: list):
        """

        :param clients_running_mean:
        :return:
        """
        global_mean = torch.zeros_like(clients_running_mean[0])
        for running_mean, client_training_size in zip(clients_running_mean, clients_training_size):
            global_mean += client_training_size * running_mean
        return global_mean / sum(clients_training_size)

    @override
    def client_local_round(self, client, round_: int, global_weight, local_parameters,
                           clients_training_size, clients_validation_size,
                           clients_training_metrics, clients_validation_metrics):
        """
        Define the process perform per client. Isolate this process in a dedicated function is usefull for parallelisation.

        :param round_:
        :param client:
        :param global_weight:
        :param local_parameters:
        :param clients_training_size:
        :param clients_validation_size:
        :param clients_training_metrics:
        :param clients_validation_metrics:
        :return:
        """
        client_weight, client_running_mean, training_data, validation_data = client.local_training(round_, global_weight)
        local_parameters[client.client_id] = client_weight
        clients_training_size[client.client_id] = training_data[0]
        clients_training_metrics[client.client_id] = training_data[1]
        clients_validation_size[client.client_id] = validation_data[0]
        clients_validation_metrics[client.client_id] = validation_data[1]

    @override
    def training_round(self, round_) -> dict:
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
                    global_weight,
                    local_parameters,
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
                self.client_local_round(client, round_, global_weight, local_parameters,
                                        clients_training_size, clients_validation_size,
                                        clients_training_metrics, clients_validation_metrics)

        new_global_weight = self.aggregate_weight(local_parameters, clients_training_size)
        self.log_server_weight(global_weight, new_global_weight, local_parameters, round_)
        self.global_model.set_parameters(new_global_weight, round_)

        training_metrics = self.aggregate_metrics_client_set(clients_training_metrics, clients_training_size)
        validation_metrics = self.aggregate_metrics_client_set(clients_validation_metrics, clients_validation_size)

        # self.average_global_mean(clients_running_mean)

        return {**training_metrics, **validation_metrics}
