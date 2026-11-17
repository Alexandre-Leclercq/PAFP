import argparse
import logging
from pathlib import Path
from omegaconf import open_dict, OmegaConf
from lightning import seed_everything
from scripts.utils.resolvers import register_all_resolvers
from scripts.utils import setup_logging, log_run_results
from scripts.utils.load_exp import find_client_checkpoint, collect_federated_checkpoints, load_config
from scripts.utils.pipeline import initialize_centralize_learning_pipeline, initialize_federated_learning_pipeline

log = logging.getLogger(__name__)

def resume_centralized(cfg, output_dir: Path, ckpt_hint: str):
    ckpt_path = find_client_checkpoint(output_dir / "checkpoints", ckpt_hint)
    log.info("Resuming centralized learning from %s", ckpt_path)

    model, datamodule, trainer = initialize_centralize_learning_pipeline(cfg)

    if hasattr(model, "weighted_loss") and hasattr(datamodule, "class_weights"):
        log.info("Setting class_weights on model")
        model.set_class_weights(datamodule.class_weights)

    # ckpt_path restores weights, optimizer, scheduler, epoch/step counters, AMP scaler
    trainer.fit(model, datamodule, ckpt_path=str(ckpt_path))
    return trainer.test(ckpt_path="best", datamodule=datamodule)

def resume_federated(cfg, output_dir: Path, ckpt_hint: str):
    ckpt_paths = collect_federated_checkpoints(output_dir, ckpt_hint)
    log.info("Resuming federated learning")

    # ckpt_paths is forwarded to initialize_federated_learning_pipeline which
    # passes each path to the corresponding Server / Client constructor.
    server = initialize_federated_learning_pipeline(cfg, mode="resume", ckpt_paths=ckpt_paths)
    server.fit()
    return server.test()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def resume_run(output_dir: Path, ckpt_hint: str = "last"):
    cfg = load_config(output_dir)

    seed_everything(cfg.seed, workers=True)

    paradigm = cfg.learning.paradigm
    log.info("Paradigm: %s", paradigm)

    if paradigm == "centralized":
        return resume_centralized(cfg, output_dir, ckpt_hint)
    elif paradigm == "federated":
        return resume_federated(cfg, output_dir, ckpt_hint)
    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a PyTorch Lightning + Hydra training run."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Hydra output directory of the run to resume. "
            "For multi-run setups pass the specific sub-run dir, e.g. outputs/.../run_2"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default="last",
        metavar="HINT",
        help=(
            "Which checkpoint to resume from. "
            "'last' (default) | 'best' | path relative to each participant's checkpoints/ dir. "
            "Server interprets 'last' as last.pt and 'best' as the highest-round global_model_round_N.pt. "
            "Clients interpret 'last' as last.ckpt and 'best' as best-*.ckpt."
        ),
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

    test_metrics = resume_run(args.output_dir, ckpt_hint=args.ckpt)

    results_path = args.output_dir / "test_results.csv"
    log_run_results(test_metrics, results_path)
    log.info("Test results saved to %s", results_path)