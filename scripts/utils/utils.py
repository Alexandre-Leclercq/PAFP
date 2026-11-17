import logging
from pathlib import Path
from typing import Mapping
import csv
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

def setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

def log_exp_comment(trainer, comment: str):
    for logger in trainer.loggers:
        if isinstance(logger, TensorBoardLogger):
            # Get the underlying SummaryWriter object
            writer = logger.experiment
            writer.add_text("exp_comment", comment)

def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} ([y]/n): ").strip().lower()
        if answer in ['y', 'yes', '']:
            return True
        elif answer in ['n', 'no']:
            return False
        print("Please respond with 'y' or 'n'.")

def log_run_results(metrics_dict: Mapping[str, float], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    # "a" opens the file in append mode; newline="" prevents extra blank rows
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        # Pass the dictionary keys as the column fieldnames
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())

        # If the file didn't exist before, write the header row first
        if not file_exists:
            writer.writeheader()

        # Append the dictionary directly as a row
        writer.writerow(metrics_dict)


