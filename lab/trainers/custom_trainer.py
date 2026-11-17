import logging
from typing import Any, Optional, Union
from weakref import proxy

import torch
from lightning import Trainer, LightningModule
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch.loggers.utilities import _log_hyperparams
from lightning.pytorch.loops.utilities import _parse_loop_limits
from lightning.pytorch.trainer import call
from lightning.pytorch.trainer.configuration_validator import _verify_loop_configurations
from lightning.pytorch.trainer.states import TrainerFn, TrainerStatus
from lightning.pytorch.utilities import parsing
from lightning.pytorch.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.seed import isolate_rng
from lightning.pytorch.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)


log = logging.getLogger(__name__)

class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ready_to_fit = False
        # print(self.model)

    def init_fit(self,
                 model: LightningModule,
                 train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
                 val_dataloaders: Optional[EVAL_DATALOADERS] = None,
                 datamodule: Optional[LightningDataModule] = None,
                 ckpt_path: Optional[_PATH] = None,
    ) -> None:
        model = _maybe_unwrap_optimized(model)
        self.strategy._lightning_module = model
        _verify_strategy_supports_compile(model, self.strategy)
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        self.should_stop = False

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloaders is not None or self.val_dataloaders is not None) and datamodule is not None:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
            )

        # links data to the trainer
        self._data_connector.attach_data(
            model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule
        )

        assert self.state.fn is not None
        ckpt_path = self._checkpoint_connector._select_ckpt_path(
            self.state.fn,
            ckpt_path,
            model_provided=True,
            model_connected=self.lightning_module is not None,
        )

        if self.state.fn == TrainerFn.FITTING:
            min_epochs, max_epochs = _parse_loop_limits(
                self.min_steps, self.max_steps, self.min_epochs, self.max_epochs, self
            )
            self.fit_loop.min_epochs = min_epochs
            self.fit_loop.max_epochs = max_epochs

        if self.barebones:
            # no progress bar in barebones can make it look like the Trainer hung
            rank_zero_info(
                "`Trainer(barebones=True)` started running. The progress bar is disabled so you might want to"
                " manually print the progress in your model."
            )

        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # attach model to the strategy
        self.strategy.connect(model)

        self._callback_connector._attach_model_callbacks()
        self._callback_connector._attach_model_logging_functions()

        _verify_loop_configurations(self)

        # ----------------------------
        # SET UP THE TRAINER
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: setting up strategy environment")
        self.strategy.setup_environment()
        self.__setup_profiler()

        log.debug(f"{self.__class__.__name__}: preparing data")
        self._data_connector.prepare_data()

        call._call_setup_hook(self)  # allow user to set up LightningModule in accelerator environment
        log.debug(f"{self.__class__.__name__}: configuring model")
        call._call_configure_model(self)

        # check if we should delay restoring checkpoint till later
        if not self.strategy.restore_checkpoint_after_setup:
            log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path)

        # reset logger connector
        self._logger_connector.reset_results()
        self._logger_connector.reset_metrics()

        # strategy will configure model and move it to the device
        self.strategy.setup(self)

        # hook
        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_start")
            call._call_lightning_module_hook(self, "on_fit_start")

        _log_hyperparams(self)

        if self.strategy.restore_checkpoint_after_setup:
            log.debug(f"{self.__class__.__name__}: restoring module and callbacks from checkpoint path: {ckpt_path}")
            self._checkpoint_connector._restore_modules_and_callbacks(ckpt_path)

        loaded = self._checkpoint_connector._loaded_checkpoint
        restored_epoch = loaded["epoch"] if loaded else 0
        self.fit_loop.max_epochs = restored_epoch
        self.fit_loop.epoch_loop.max_steps = 0

        # restore optimizers, etc.
        log.debug(f"{self.__class__.__name__}: restoring training state")
        self._checkpoint_connector.restore_training_state()

        self._checkpoint_connector.resume_end()
        self._signal_connector.register_signal_handlers()

        self.fit_loop.max_epochs = restored_epoch
        self.fit_loop.epoch_loop.max_steps = 0

        # Reset epoch progress so the loop doesn't think it has already
        # completed the restored epoch. Without this, fit_loop.run() returns
        # immediately because epoch_progress.current.completed == max_epochs.
        if loaded is not None:
            self.fit_loop.epoch_progress.reset()
            self.fit_loop.epoch_progress.current.completed = restored_epoch
            self.fit_loop.epoch_progress.total.completed = restored_epoch

        self.ready_to_fit = True

    def partial_fit_epochs(
            self,
            epochs: int,
    ):
        assert self.ready_to_fit

        self.fit_loop.max_epochs += epochs
        self.fit_loop.epoch_loop.max_steps = -1

        call._call_and_handle_interrupt(
            self, self._partial_fit_impl
        )

    def partial_fit_iterations(
            self,
            iterations: int,
    ):
        assert self.ready_to_fit

        self.fit_loop.max_epochs = -1
        self.fit_loop.epoch_loop.max_steps += iterations

        call._call_and_handle_interrupt(
            self, self._partial_fit_impl
        )


    def _partial_fit_impl(self) -> None:
        log.debug(f"{self.__class__.__name__}: trainer partial fit stage")

        # wait for all to join if on distributed
        self.strategy.barrier("run-stage")
        self.lightning_module.zero_grad()

        if self.training:
            with isolate_rng():
                self._run_sanity_check()
            with torch.autograd.set_detect_anomaly(self._detect_anomaly):
                self.fit_loop.run()
            return None
        raise RuntimeError(f"Unexpected state {self.state}")

    def complete_fit(self):
        # ----------------------------
        # POST-Training CLEAN UP
        # ----------------------------
        log.debug(f"{self.__class__.__name__}: trainer tearing down")
        self._teardown()

        if self.state.fn == TrainerFn.FITTING:
            call._call_callback_hooks(self, "on_fit_end")
            call._call_lightning_module_hook(self, "on_fit_end")

        log.debug(f"{self.__class__.__name__}: calling teardown hooks")
        call._call_teardown_hook(self)

        self.state.status = TrainerStatus.FINISHED
        self.state.stage = None

        assert self.state.stopped
        self.training = False
        self.ready_to_fit = False

    def __setup_profiler(self) -> None:
        assert self.state.fn is not None
        local_rank = self.local_rank if self.world_size > 1 else None
        self.profiler._lightning_module = proxy(self.lightning_module)
        self.profiler.setup(stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir)
