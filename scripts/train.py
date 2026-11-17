import os
from pathlib import Path
import pandas as pd
import hydra
import logging
import copy
from omegaconf import DictConfig, OmegaConf, open_dict
from lightning import seed_everything
from scripts.utils import setup_logging, ask_yes_no, log_exp_comment, log_run_results
from scripts.utils.pipeline import initialize_centralize_learning_pipeline, initialize_federated_learning_pipeline
from scripts.utils.resolvers import register_all_resolvers

def centralized_learning_pipeline(cfg, logger, exp_comment = None):
    print('centralized learning pipeline')

    model, datamodule, trainer = initialize_centralize_learning_pipeline(cfg)
    if exp_comment is not None:
        log_exp_comment(trainer, exp_comment)

    if hasattr(model, 'weighted_loss') and hasattr(datamodule, 'class_weights'):
        print('setting class_weights to model')
        model.set_class_weights(datamodule.class_weights)
    trainer.fit(model, datamodule)
    return trainer.test(ckpt_path="best", datamodule=datamodule)

def federated_learning_pipeline(cfg, logger, exp_comment = None):
    server = initialize_federated_learning_pipeline(cfg, mode='train')
    if exp_comment is not None:
        log_exp_comment(server.global_trainer, exp_comment)

    server.fit()
    print('Federated Learning Finished')
    return server.test()

def run_orchestrator(cfg, logger, exp_comment = None):
    if cfg.learning.paradigm == 'centralized':  # centralized learning
        test_metrics = centralized_learning_pipeline(cfg, logger, exp_comment)
    elif cfg.learning.paradigm == 'federated':  # federated learning case
        test_metrics = federated_learning_pipeline(cfg, logger, exp_comment)
    else:
        raise ValueError(f'unknown paradigm {cfg.learning.paradigm}')
    return test_metrics


register_all_resolvers()
@hydra.main(version_base=None, config_path='../configs', config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.verbose)
    logger = logging.getLogger(__name__)

    if not cfg.code_test and cfg.interactive:
        logger.info(OmegaConf.to_yaml(cfg))

        valid = ask_yes_no('Is this the intended config?')
        if not valid:
            print('Invalid config aborting script\'s execution')
            return

    n_runs = OmegaConf.select(cfg, "n_runs", default=None)

    if n_runs is None:
        raise KeyError(f'Configuration missing mandatory key "n_runs"')
    assert n_runs >= 1, f'n_runs should be greater than 1, got {n_runs}'

    if not cfg.code_test and cfg.interactive:
        comment_exp = input(f"experience comment: ").strip().lower()
    else:
        comment_exp = None

    hydra_context = hydra.core.hydra_config.HydraConfig.get()
    hydra_run_dir = Path(hydra_context.runtime.output_dir)

    if n_runs == 1:
        test_metrics = run_orchestrator(cfg, logger, comment_exp)
        log_run_results(test_metrics, hydra_run_dir / "results.csv")
    else:
        for run_idx in range(cfg.n_runs):
            seed = cfg.seed + run_idx
            print(f'starting Run #{run_idx + 1}/{cfg.n_runs} with seed {seed}')

            seed_everything(seed, workers=True)

            run_cfg = copy.deepcopy(cfg)

            with open_dict(run_cfg):
                # Update base_dir to: ./outputs/YYYY-MM-DD_HH-MM-SS/run_0, run_1, etc.
                new_path = os.path.join(run_cfg.base_dir_run, f"run_{run_idx}")
                OmegaConf.update(run_cfg, "base_dir_run", new_path)
                OmegaConf.update(run_cfg, "seed", seed)

            comment_exp_run = f'run_{run_idx} : {comment_exp}' if comment_exp is not None else None

            test_metrics = run_orchestrator(run_cfg, logger, comment_exp_run)

            test_metrics = {"run": run_idx, **test_metrics}
            log_run_results(test_metrics, hydra_run_dir/"results.csv")
        else:
            df_metrics = pd.read_csv(hydra_run_dir/"results.csv")
            mean_series = df_metrics.mean(numeric_only=True)
            std_series = df_metrics.std(numeric_only=True)
            mean_row = pd.DataFrame([mean_series], index=["mean"])
            std_row = pd.DataFrame([std_series], index=["std"])
            df_with_stats = pd.concat([df_metrics, mean_row, std_row], axis="rows")
            df_with_stats.to_csv(hydra_run_dir/ "test_metrics_processed.csv")

if __name__ == "__main__":
    main()
