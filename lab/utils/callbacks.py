import os
import torch
import h5py
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import Callback
from typing import Any, Optional


class LatentSaverCallback(Callback):
    """
    Saves the latent representation of the training set every N epochs.

    This callback saves latents progressively to an HDF5 file to avoid
    storing the entire dataset in memory.
    """

    def __init__(self, output_dir: str, save_every_n_epochs: int = 1, prefix_name: Optional[str] = None):
        super().__init__()
        self.output_dir = output_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.prefix_name = prefix_name
        os.makedirs(self.output_dir, exist_ok=True)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called when the train epoch begins."""
        # Check if we need to save on this epoch
        if (trainer.current_epoch % self.save_every_n_epochs) == 0:
            prefix = f'{self.prefix_name}_' if self.prefix_name else ''
            filepath = os.path.join(self.output_dir, f"{prefix}latents_epoch_{trainer.current_epoch}.h5")
            print(f"\nEpoch {trainer.current_epoch}: Saving latents & targets to {filepath}")
            # Open HDF5 file in write mode
            self.h5_file = h5py.File(filepath, 'w')
            # We don't know the dataset size yet, so we'll create it on the first batch
            self.latents_dataset = None
            self.targets_dataset = None
        else:
            self.h5_file = None
            self.latents_dataset = None
            self.targets_dataset = None

    def on_train_batch_end(
            self,
            trainer: L.Trainer,
            pl_module: L.LightningModule,
            outputs: dict,
            batch: Any,
            batch_idx: int
    ):
        """Called when the train batch ends."""
        # Check if we are in a "saving" epoch
        if self.h5_file is None:
            return

        # 'outputs' is the dictionary returned from training_step
        if 'latents' not in outputs or 'targets' not in outputs:
            raise KeyError("The `training_step` must return a dictionary with 'latents' and 'targets' keys.")

        # Move latents to CPU and convert to numpy
        latents_np = outputs['latents'].detach().cpu().numpy()
        targets_np = outputs['targets'].detach().cpu().numpy()

        # Ensure batch dimension exists
        if latents_np.ndim == 1:
            latents_np = np.expand_dims(latents_np, axis=0)
        if targets_np.ndim == 0:  # Handle scalar targets if batch size is 1
            targets_np = np.expand_dims(targets_np, axis=0)

        batch_size = latents_np.shape[0]

        # 2. Initialize Datasets on the first batch
        if self.latents_dataset is None:
            # Create Latents Dataset
            latent_dim = latents_np.shape[1]
            self.latents_dataset = self.h5_file.create_dataset(
                'latents',
                shape=(0, latent_dim),
                maxshape=(None, latent_dim),
                dtype='f4',
                chunks=True
            )

            # Create Targets Dataset
            # target_shape handles both 1D labels (e.g., shape (32,)) and 2D targets
            target_shape = targets_np.shape[1:]
            self.targets_dataset = self.h5_file.create_dataset(
                'targets',
                shape=(0, *target_shape),
                maxshape=(None, *target_shape),
                dtype=targets_np.dtype,  # Usually int64
                chunks=True
            )


        # Append the new batch of latents
        current_size = self.latents_dataset.shape[0]

        # Resize both datasets
        self.latents_dataset.resize(current_size + batch_size, axis=0)
        self.targets_dataset.resize(current_size + batch_size, axis=0)

        # Insert the new batch
        self.latents_dataset[current_size:] = latents_np
        self.targets_dataset[current_size:] = targets_np

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called when the train epoch ends."""
        # Close the HDF5 file if it was opened
        if self.h5_file is not None:
            print(f"Finished saving latents for epoch {trainer.current_epoch}. Final shape: {self.latents_dataset.shape}")
            self.h5_file.close()
            self.h5_file = None
            self.latents_dataset = None
            self.targets_dataset = None