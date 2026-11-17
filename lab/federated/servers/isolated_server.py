from lab.federated.servers import BasicFederatedServer
from lab.federated.clients import Client
from lab.models import BaseFedModel
from lightning.pytorch import Trainer
from random import sample
import torch
import threading

"""
This is simulated Federated Learning.
We use the server as a scheduler.
This server represent a realistic scenario where no FL is used.
Thus, each client are isolated and doing local training without sharing their model.
"""

class IsolatedServer(BasicFederatedServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.global_model = None

    def sample_clients(self):
        """
        No sampling for this scenario
        :return:
        """
        return self.clients

    def aggregate_weight(self, local_weight, local_size):
        """
        No weight aggregation for this scenario
        :param local_weight:
        :param local_size:
        :return:
        """
        pass

    def client_local_round(self, client, round_: int,
                           clients_training_size, clients_validation_size,
                           clients_training_metrics, clients_validation_metrics):
        """
        Define the process perform per client. Isolate this process in a dedicated function is usefull for parallelization.

        :param client:
        :param round_:
        :param clients_training_size:
        :param clients_validation_size:
        :param clients_training_metrics:
        :param clients_validation_metrics:
        :return:
        """
        training_data, validation_data = client.local_training(round_)
        clients_training_size[client.client_id] = training_data[0]
        clients_training_metrics[client.client_id] = training_data[1]
        clients_validation_size[client.client_id] = validation_data[0]
        clients_validation_metrics[client.client_id] = validation_data[1]

    def training_round(self, round_: int) -> dict:
        """
        perform one training round.
        - send global weight to clients
        - perform local training on clients
        - receive clients updates
        - aggregate clients weight to global weight
        - compute global metrics

        :return: a dict of global metrics (round_metrics) which are computed on local train and val set for each client
        """
        clients_training_size = [None] * len(self.clients)
        clients_validation_size = [None] * len(self.clients)
        clients_training_metrics = [None] * len(self.clients)
        clients_validation_metrics = [None] * len(self.clients)

        threads = []

        if self.parallelism:  # parallelization case
            for client in self.clients:
                t = threading.Thread(target=self.client_local_round, args=(
                        client,
                        round_,
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
            for client in self.clients:
                self.client_local_round(client, round_,
                                        clients_training_size, clients_validation_size,
                                        clients_training_metrics, clients_validation_metrics)

        training_metrics = self.aggregate_metrics_client_set(clients_training_metrics, clients_training_size)
        validation_metrics = self.aggregate_metrics_client_set(clients_validation_metrics, clients_validation_size)

        return {**training_metrics, **validation_metrics}
