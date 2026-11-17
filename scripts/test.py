import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from lightning import seed_everything
from scripts.utils.resolvers import register_all_resolvers
from scripts.utils import setup_logging, log_run_results
from scripts.utils.load_exp import find_client_checkpoint, collect_federated_checkpoints, load_config
from scripts.utils.pipeline import initialize_centralize_learning_pipeline, initialize_federated_learning_pipeline

log = logging.getLogger(__name__)

def test_centralized(cfg, output_dir: Path, ckpt_hint: str) -> dict:
    ckpt_path = find_client_checkpoint(output_dir / "checkpoints", ckpt_hint)
    log.info("Testing centralized model from %s", ckpt_path)

    model, datamodule, trainer = initialize_centralize_learning_pipeline(cfg)

    # Mirrors exactly what train.py does after trainer.fit():
    #   trainer.test(ckpt_path="best", datamodule=datamodule)
    # Here we pass the resolved path explicitly instead of "best" so the
    # script honours the --ckpt argument.
    return trainer.test(model, datamodule=datamodule, ckpt_path=str(ckpt_path))

def test_federated(cfg, output_dir: Path, ckpt_hint: str) -> dict:
    ckpt_paths = collect_federated_checkpoints(output_dir, ckpt_hint)
    log.info("Testing federated system")

    # Clients are built in mode='test': datamodule.setup(stage='test') is called,
    # no init_fit is needed. The server's evaluate_global_model('test') will load
    # the server checkpoint from global_weights_path; we pass the resolved path
    # explicitly so it uses the one matching --ckpt rather than picking arbitrarily.
    server = initialize_federated_learning_pipeline(cfg, mode="test", ckpt_paths=ckpt_paths)

    # server.test() replicates exactly what train.py calls at the end of federated training:
    #   - each client.test(ckpt_path=<hint>) on their local test set
    #   - server.evaluate_global_model('test') on all client test sets
    return server.test(mode=ckpt_hint)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_run(output_dir: Path, ckpt_hint: str = "best") -> dict:
    cfg = load_config(output_dir)

    seed_everything(cfg.seed, workers=True)

    paradigm = cfg.learning.paradigm
    log.info("Paradigm: %s", paradigm)

    if paradigm == "centralized":
        return test_centralized(cfg, output_dir, ckpt_hint)
    elif paradigm == "federated":
        return test_federated(cfg, output_dir, ckpt_hint)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a PyTorch Lightning + Hydra training run from its output directory."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Hydra output directory of the run to test. "
            "For multi-run setups pass the specific sub-run dir, e.g. outputs/.../run_2"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default="best",
        metavar="HINT",
        help=(
            "Which checkpoint to test. "
            "'best' (default) | 'last' | path relative to each participant's checkpoints/ dir. "
            "Server: 'best' picks the highest-round global_model_round_N.pt; "
            "'last' picks last.pt. "
            "Clients: 'best' picks best-*.ckpt; 'last' picks last.ckpt."
        ),
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test metrics to <output_dir>/test_results.csv",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    register_all_resolvers()

    args = parse_args()
    setup_logging(verbose=(args.log_level == "DEBUG"))
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.output_dir.exists():
        raise SystemExit(f"Output directory does not exist: {args.output_dir}")

    test_metrics = test_run(args.output_dir, ckpt_hint=args.ckpt)

    if args.save_results:
        results_path = args.output_dir / "test_results.csv"
        log_run_results(test_metrics, results_path)
        log.info("Test results saved to %s", results_path)