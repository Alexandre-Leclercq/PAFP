from hydra.utils import instantiate
from lab.data.partitioning import partition_data
from lab.utils import instantiate_from_class_path
import lightning as L
import torch
import os
from typing import Literal, Optional
from omegaconf import OmegaConf

def initialize_centralize_learning_pipeline(cfg):
    data_module = instantiate(cfg.learning.data.datamodule, reduce_mode=cfg.code_test)
    cfg.learning.model.n_class = data_module.get_n_class()
    model = instantiate(cfg.learning.model)
    cfg.learning.trainer.callbacks = list(cfg.learning.trainer.callbacks.values())
    trainer = instantiate(cfg.learning.trainer)
    return model, data_module, trainer


def initialize_federated_learning_pipeline(cfg, mode: Literal['train', 'resume', 'test'], ckpt_paths:Optional[dict]=None):
    # implement a partitioning method from a specific dataset
    # each partition is then provided to its corresponding client
    # client constructor then initiate a datamodule from the given dataset

    # loop for each client to construct each client
    clients = []

    n_client = cfg.learning.clients.n_client
    if cfg.parallelism:
        num_gpus = torch.cuda.device_count()
        assert num_gpus >= n_client  # assert to have enough gpu for the number of client

    # Retrieve the clients and server config and interpolate variables
    clients_cfg = OmegaConf.to_container(cfg.learning.clients, resolve=True)
    server_cfg = OmegaConf.to_container(cfg.learning.server, resolve=True)

    """
    1. Datamodule instantiation
    """
    if cfg.learning.partitioning.method == 'multi_dataset':
        print('multi dataset case')
        assert len(cfg.learning.partitioning.data) == n_client
        datas_config = list(cfg.learning.partitioning.data.values())
        datamodules = []
        for data_config in datas_config:
            datamodule_config = OmegaConf.merge(data_config.datamodule, cfg.learning.partitioning.datamodule_shared_parameters)
            datamodules.append(instantiate(datamodule_config, reduce_mode=cfg.code_test))
    else:  # mono dataset w/ a partitioning method
        datamodules: list[L.LightningDataModule] = partition_data(
            dataset=cfg.learning.partitioning.data.name,
            dataset_dir=cfg.learning.partitioning.data.datamodule.dataset_path,
            method=cfg.learning.partitioning.method,
            n_partitions=n_client,
            reduce_mode=cfg.code_test,
            config_datamodule=cfg.learning.partitioning.data.datamodule,
            partitioning_parameters=cfg.learning.partitioning.parameters
        )

    """
    2. Clients instantiation
    """
    # get the clients config dict and remove the key that we manually set in the instantiate loop
    # such that clients_cfg contain only keys unique to the specific method used.
    print(f'cfg.learning.clients:')
    clients_cfg.pop("class_path", None)
    model_cfg = clients_cfg.pop("model", None)
    trainer_cfg = clients_cfg.pop("trainer", None)
    max_steps = clients_cfg.pop("max_steps", None)
    clients_cfg.pop("n_client", None)
    gpu_index = cfg.gpu_index

    for i in range(n_client):  # loop to instantiate each client
        model = instantiate(model_cfg)
        # dynamically allocate worker depending on client number
        datamodules[i].set_n_worker(os.cpu_count() // n_client)
        if cfg.parallelism:
            assert len(gpu_index) >= n_client
            device_id = gpu_index[i]
        else:
            device_id = gpu_index[0]
        clients.append(instantiate_from_class_path(
            cfg.learning.clients.class_path,
            dict(  # parameters of Basic Client (parent class of all client)
                client_id=i,
                device_id=device_id,
                model=model,
                datamodule=datamodules[i],
                robust_evaluation=cfg.learning.server.robust_evaluation,
                mode=mode,
                ckpt_path=ckpt_paths.get(f'client_{i}') if ckpt_paths else None,
                trainer_cfg=OmegaConf.create(trainer_cfg.copy()),
                max_steps=max_steps[i] if isinstance(max_steps, list) else max_steps,
                **clients_cfg  # clients parameters unique to the used method (not present in the Basic Client)
            ),
        ))

    """
    3. Server instantiation
    """
    # retrieve the server config and drop key that will manually be handled.
    server_cfg.pop("class_path", None)  # path to instantiate server
    server_cfg.pop("global_model", None)  # global model already instantiate
    server_cfg.pop("global_trainer", None)  # already instantiate
    server_cfg = OmegaConf.create(server_cfg)

    global_model = instantiate(cfg.learning.server.global_model)

    server = instantiate_from_class_path(
        cfg.learning.server.class_path,
        dict(
            clients=clients,
            global_model=global_model,
            global_trainer_cfg=cfg.learning.server.global_trainer,
            ckpt_path=ckpt_paths.get('server') if ckpt_paths else None,
            save_top_k=cfg.learning.clients.trainer.callbacks.model_checkpoint.save_top_k,
            **server_cfg  # remaining parameter in the server config
        )
    )
    print('server initialized')
    return server