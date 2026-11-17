from lab.federated.servers.fedavg_server import FedAvgServer


class DittoServer(FedAvgServer):
    """
    Ditto: aggregation is identical to FedAvg (weighted average of the global model's weights).
    Personalization lives entirely client-side in DittoModel.
    """
    pass
