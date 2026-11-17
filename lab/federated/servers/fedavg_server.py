from lab.federated.servers import BasicFederatedServer
import torch
import threading

"""
This is simulated Federated Learning.
We use the server as a scheduler
"""


class FedAvgServer(BasicFederatedServer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_weight(self, local_weight: list[dict], local_size: list[int]):
        """
        Aggregate local and global weights.
        :param local_weight:
        :param local_size:
        :return: the new global_weight
        """
        avg_weight = local_weight[0].copy()
        total_size = sum(local_size)
        for key in avg_weight.keys():
            # every aggregation will be performed in float32
            # this ensures to reduce approximation error
            # while avoiding mismatches type (for parameters originally stored as integer)
            avg_weight[key] = torch.zeros_like(avg_weight[key], dtype=torch.float32)
            for i in range(len(local_weight)):
                avg_weight[key] += local_size[i]/total_size * local_weight[i][key].float()

        # then we cast back every tensor to its original type
        for key in avg_weight.keys():
            original_dtype = local_weight[0][key].dtype
            avg_weight[key] = avg_weight[key].to(dtype=original_dtype)

        return avg_weight

    def client_local_round(self, client, round_: int, global_weight, local_parameters,
                           clients_training_size, clients_validation_size,
                           clients_training_metrics, clients_validation_metrics):
        """
        Define the process perform per client. Isolate this process in a dedicated function is usefull for parallelization.

        :param client:
        :param round_:
        :param global_weight:
        :param local_parameters:
        :param clients_training_size:
        :param clients_validation_size:
        :param clients_training_metrics:
        :param clients_validation_metrics:
        :return:
        """
        print(f'server start training of client {client.client_name}')
        client_weight, training_data, validation_data = client.local_training(round_, global_weight)
        print(f'server end training of client {client.client_name}')
        local_parameters[client.client_id] = client_weight
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
        sampled_clients = self.sample_clients()

        local_parameters = [None] * len(sampled_clients)
        clients_training_size = [None] * len(sampled_clients)
        clients_validation_size = [None] * len(sampled_clients)
        clients_training_metrics = [None] * len(sampled_clients)
        clients_validation_metrics = [None] * len(sampled_clients)
        global_weight = self.global_model.get_parameters()

        if self.parallelism:  # parallelisation case
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
        print(f'server aggregate weight')
        new_global_weight = self.aggregate_weight(local_parameters, clients_training_size)
        self.log_server_weight(global_weight, new_global_weight, local_parameters, round_)
        self.global_model.set_parameters(new_global_weight, round_)
        print(f'server aggregate metrics')
        training_metrics = self.aggregate_metrics_client_set(clients_training_metrics, clients_training_size)
        validation_metrics = self.aggregate_metrics_client_set(clients_validation_metrics, clients_validation_size)
        print(f'server return metrics')
        return {**training_metrics, **validation_metrics}
