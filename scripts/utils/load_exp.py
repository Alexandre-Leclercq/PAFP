import logging
from pathlib import Path
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

def find_server_checkpoint(ckpt_dir: Path, ckpt_hint: str) -> Path:
    """
    Resolve a server checkpoint (.pt) from the hint.
      - last  → last.pt  (written every round by save_last_global_weights)
      - best  → global_model_round_N.pt with the highest round number among saved top-k
    """
    if ckpt_hint == "last":
        path = ckpt_dir / "last.pt"
        if not path.exists():
            raise FileNotFoundError(f"last.pt not found in {ckpt_dir}.")
        return path

    if ckpt_hint == "best":
        candidates = sorted(ckpt_dir.glob("global_model_round_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No best server checkpoint found in {ckpt_dir}")
        return max(candidates, key=lambda p: int(p.stem.split("_")[-1]))

    path = Path(ckpt_hint)
    if not path.is_absolute():
        path = ckpt_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Server checkpoint not found: {path}")
    return path

def find_client_checkpoint(ckpt_dir: Path, ckpt_hint: str) -> Path:
    """
    Resolve a client checkpoint (.ckpt) from the hint.
    """
    if ckpt_hint == "best":
        candidates = sorted(ckpt_dir.glob("epoch=*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No best checkpoint in {ckpt_dir}")
        return candidates[0]

    if ckpt_hint == "last":
        path = ckpt_dir / "last.ckpt"
        if not path.exists():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            path = ckpts[-1]
            log.warning("last.ckpt not found, falling back to most recent: %s", path)
        return path

    path = Path(ckpt_hint)
    if not path.is_absolute():
        path = ckpt_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Client checkpoint not found: {path}")
    return path

def collect_federated_checkpoints(output_dir: Path, ckpt_hint: str) -> dict[str, Path]:
    """
    Walk the federated output directory and collect one checkpoint per participant.

    Server checkpoints are .pt files; client checkpoints are .ckpt files.
    Returns a dict like:
        {
            "server":   Path(".../server/checkpoints/last.pt"),
            "client_0": Path(".../client_0/checkpoints/last.ckpt"),
            "client_1": Path(".../client_1/checkpoints/last.ckpt"),
            ...
        }
    """
    ckpt_paths: dict[str, Path] = {}

    server_ckpt_dir = output_dir / "server" / "checkpoints"
    if server_ckpt_dir.exists():
        ckpt_paths["server"] = find_server_checkpoint(server_ckpt_dir, ckpt_hint)
    else:
        log.warning("No server checkpoint directory found at %s", server_ckpt_dir)

    for client_dir in sorted(output_dir.glob("client_*")):
        client_ckpt_dir = client_dir / "checkpoints"
        if not client_ckpt_dir.exists():
            log.warning("No checkpoints directory found for %s, skipping", client_dir.name)
            continue
        ckpt_paths[client_dir.name] = find_client_checkpoint(client_ckpt_dir, ckpt_hint)


    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoint directories found under {output_dir}. "
            "Expected server/ and client_*/ subdirectories."
        )

    log.info("Collected federated checkpoints:")
    for name, path in ckpt_paths.items():
        log.info("  %-12s → %s", name, path)

    return ckpt_paths

def load_config(output_dir: Path) -> OmegaConf:
    config_path = output_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Hydra config not found at {config_path}.\n"
            "Make sure <output_dir> points to a directory that contains .hydra/."
        )
    cfg = OmegaConf.load(config_path)
    log.info("Loaded config from %s", config_path)
    return cfg