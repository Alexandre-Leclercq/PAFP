from .basic_client import Client


class FedAvgClient(Client):
    """
    Useless for know but will be usefull when some specific element of FedAvg
    wouldn't be needed to be in the client template class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

