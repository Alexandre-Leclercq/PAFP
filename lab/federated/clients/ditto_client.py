from .basic_client import Client
from lab.models import DittoModel


class DittoClient(Client):
    def __init__(self, lambda_reg, model, *args, **kwargs):
        model = DittoModel(base_model=model, lambda_reg=lambda_reg)
        super().__init__(model=model, *args, **kwargs)
