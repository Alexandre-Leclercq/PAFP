from typing import TYPE_CHECKING
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
from matplotlib import pyplot as plt
import io

if TYPE_CHECKING:  # only do the import during the type_checking and not the runtime to avoid circular import
    from lab.data.datamodules import BaseDataModule
    from lab.federated.clients import Client


def client_log_pretty_dataset_distribution(writer, datamodule: 'BaseDataModule') -> None:
    """
    log a pretty table of the dataset distribution
    :param writer:
    :param datamodule:
    :return:
    """

    table_html = "<table border='1'><tr><th>Dataset</th>"
    for i in range(datamodule.get_n_class()):
        table_html += f"<th>Class {i}</th>"
    table_html += "<th>n</th></tr>"

    set_class_distrib = {
        'train': datamodule.get_train_class_distribution(),
        'val': datamodule.get_val_class_distribution(),
        'test': datamodule.get_test_class_distribution()
    }

    set_count = {
        'train': datamodule.get_len_train_set(),
        'val': datamodule.get_len_val_set(),
        'test': datamodule.get_len_test_set()
    }

    for set in ['train', 'val', 'test']:
        table_html += f"<tr><td>{set}</td>"
        for class_proportion in set_class_distrib[set]:
            table_html += f"<td>{class_proportion.item()}</td>"
        table_html += f"<td>{set_count[set]}</td></tr>"
    table_html += "</table>"

    writer.add_text("dataset_distribution", table_html, global_step=0)


def server_log_pretty_dataset_distribution(writer, clients: 'list[Client]') -> None:
    """
    log a pretty table of the dataset distribution
    :param writer:
    :param clients:
    :return:
    """
    # Build the column
    client = clients[0]
    table_html = "<table border='1'><tr><th>client</th>"
    for i in range(client.datamodule.get_n_class()):
        table_html += f"<th>Class {i}</th>"
    table_html += "<th>n</th></tr>"

    for client in clients:
        table_html += f"<tr><td>{client.client_id}</td>"
        class_proportions = client.datamodule.get_train_class_distribution()
        for class_proportion in class_proportions:
            table_html += f"<td>{class_proportion.item()}</td>"
        table_html += f"<td>{client.datamodule.get_len_train_set()}</td></tr>"
    table_html += "</table>"

    writer.add_text("federated_distribution", table_html, global_step=0)

def plot_to_image(figure):
    """Converts a matplotlib figure to a PNG image for TensorBoard."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # Decode the buffer into a numpy array/tensor
    image = plt.imread(buf)
    # Remove alpha channel if present and transpose to (C, H, W)
    image = image[:, :, :3].transpose(2, 0, 1)
    return torch.from_numpy(image)

def server_log_param_cosine_similarity(writer, n_clients: int, clients_parameters: list, new_global_weight, step: int) -> None:
    for key in clients_parameters[0].keys():  # assuming each client share the same parameters
        flattened_parameters = [clients_parameters[i][key].float().flatten() for i in range(n_clients)]
        flattened_parameters += [new_global_weight[key].float().flatten()]
        cos_sim_matrix = cosine_similarity(flattened_parameters)
        mask = np.triu(np.ones_like(cos_sim_matrix, dtype=bool))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cos_sim_matrix,
            annot=True,
            xticklabels=[f'client_{i}' for i in range(n_clients)]+['server'],
            yticklabels=[f'client_{i}' for i in range(n_clients)]+['server'],
            mask=mask,
            ax=ax,
            cmap='YlGnBu',
            fmt=".4f"
            # vmin=0, vmax=1  # Cosine similarity usually stays between 0-1 for positive data
        )
        ax.set_title(f'Similarity Matrix - Round {step}')

        image_tensor = plot_to_image(fig)
        writer.add_image(f'cosine_similarity_param/{key}', image_tensor, global_step=step)

def model_log_pretty_class_weight(writer, class_weights: torch.Tensor):
    table_html = "<table border='1'><tr>"
    for i in range(len(class_weights)):
        table_html += f"<th>Class {i}</th>"
    table_html += "</tr>"

    table_html += f"<tr>"
    for class_weight in class_weights:
        table_html += f"<td>{class_weight.item():.2f}</td>"
    table_html += "</tr></table>"

    writer.add_text("class_weight", table_html, global_step=0)
