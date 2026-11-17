# PAFP: Personalized Aligner Federated Pretrained

Official implementation of **"Federated Transfer Learning for Medical Image Classification under Mixed Skew"**,
Alexandre G. Leclercq, Sébastien Bougleux, Aurélien Corroyer-Dulmont, Romain Hérault. *FLTA 2026*.

📄 Preprint: [arXiv:2611.05035](https://arxiv.org/abs/2611.05035) (the IEEE FLTA 2026 proceedings version is forthcoming)

PAFP is a personalized federated learning method built on a **pre-trained backbone**. Each client learns lightweight
**feature alignment modules** that realign its hidden features toward a server-aggregated target distribution, which makes
the method robust to the *mixed* (label, feature and quantity) skews of multicenter medical data. On the FedISIC benchmark,
PAFP achieves the highest unweighted balanced accuracy and reduces the inter-client variance of performance by about 6x
compared to the strongest baseline.

## Project structure

    PAFP/
    ├── lab/                        # library code
    │   ├── data/                   # Fed-ISIC2019 dataset, datamodule and partitioning
    │   ├── federated/
    │   │   ├── clients/            # Isolated, FedAvg, DBE, Ditto, PAFP clients
    │   │   └── servers/            # matching servers
    │   ├── models/                 # EfficientNet, partially fine-tuned EfficientNet (ft-2B), DBE, Ditto
    │   │   ├── pafp/               # PAFP model
    │   │   └── modules/            # feature aligner, EMA, ...
    │   ├── trainers/               # custom Lightning trainer
    │   └── utils/                  # losses, callbacks, logging
    │
    ├── configs/                    # Hydra configs
    │   ├── *.yaml                  # one file per experiment (see table below)
    │   ├── data/
    │   ├── global/                 # machine-specific settings (paths, GPUs, seed, number of runs)
    │   ├── learning/               # centralized / federated algorithms and partitioning
    │   ├── model/
    │   └── trainer/
    │
    ├── scripts/                    # entry points
    │   ├── train.py
    │   ├── resume.py
    │   └── test.py
    │
    ├── pyproject.toml
    └── uv.lock

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/). PyTorch is installed from the CUDA 12.8 wheel index
(see `pyproject.toml`).

```shell
uv sync
```

## Data and pretrained weights

### Fed-ISIC2019

We use the Fed-ISIC2019 dataset from the [FLamby](https://github.com/owkin/FLamby) benchmark (6 clients).
To download and preprocess it, follow the
[FLamby Fed-ISIC2019 instructions](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019).
Then place the result so that the following layout exists:

    <dataset_dir>/ISIC2019/
    ├── train_test_split.csv
    └── ISIC_2019_Training_Input_preprocessed/

### Pretrained backbone

The backbone is timm's `efficientnet_b0.ra_in1k`. Weights are loaded from a local file, not downloaded at runtime:
download `model.safetensors` from [timm/efficientnet_b0.ra_in1k](https://huggingface.co/timm/efficientnet_b0.ra_in1k)
and save it as:

    <timm_dir>/efficientnet_b0.ra_in1k.safetensors

### Paths and machine settings

`dataset_dir`, `timm_dir`, `gpu_index`, `seed` and `n_runs` are defined in `configs/global/local_settings.yaml`
(defaults: `../dataset`, `../timm`, GPU `0`, seed `23`, `5` runs). To use another machine, edit this file or add your own
file in `configs/global/` and select it with `global=<your_settings>`.

## Reproducing the paper

Every experiment is a Hydra config in `configs/`:

```shell
uv run python -m scripts.train -cn <config_name>
```

With the default settings, each config is run 5 times (`n_runs: 5`). The paper reports the mean ± std over these runs.

| Method (paper)   | Config                                                                  |
|------------------|-------------------------------------------------------------------------|
| Isolated         | `isolated_isic2019`                                                     |
| FedAvg-ft        | `fedavg_epoch_isic2019`                                                 |
| DBE-ft           | `dbe_finetuned_epoch_isic2019`                                          |
| Ditto-ft         | `ditto_finetuned_epoch_isic2019`                                        |
| ft-2B            | `pfp_isic2019_f_b5-b6_wo_plogits`                                       |
| ft-2B + LC       | `pfp_isic2019_f_b5-b6_w_plogits`                                        |
| **PAFP**         | `pafp_isic2019_gstat_ema_b5-b6_10_lambda_mean-var`                      |
| PAFP + LC        | `pafp_isic2019_gstat_ema_b5-b6_10_lambda_mean-var_w_plogits`            |
| PAFP-alt (r=2)   | `pafp_isic2019_alternate_opt_2r_gstat_ema_b5-b6_10_lambda_mean-var`     |
| **PAFP-alt (r=10)** | `pafp_isic2019_alternate_opt_10r_gstat_ema_b5-b6_10_lambda_mean-var` |
| PAFP-alt (r=50)  | `pafp_isic2019_alternate_opt_50r_gstat_ema_b5-b6_10_lambda_mean-var`    |

*LC = Logit Corrector (`personalized_logits: True`); r = alternation interval (`switch_n_rounds`).*

The other configs in `configs/` are additional experiments that are not reported in the paper. They include other
values of λ (`*_20_lambda_*`, `*_1_lambda_*`), alternation every 20 rounds, a joint two-optimizer variant (`*_2opt_*`),
fine-tuning blocks 4–6 (`*_b4-b6_*`), and DBE trained from scratch or with a frozen backbone.

### Overriding parameters

Any value can be overridden from the command line with the usual Hydra syntax, for example:

```shell
uv run python -m scripts.train -cn pafp_isic2019_gstat_ema_b5-b6_10_lambda_mean-var n_runs=1 learning.server.n_round=100
```

To swap a config group, use `group_path@position.in.config=option`. If the group path matches its position in the
config, write only the group path, for example `learning/partitioning/data=fed_isic2019`.

## Outputs, resuming and testing

Each run writes its logs, checkpoints and the resolved config to `outputs/<YYYY-MM-DD_HH-MM-SS>/`. With several runs,
each run gets its own sub-directory (e.g. `outputs/.../run_2`).

Resume an interrupted run:

```shell
uv run python -m scripts.resume outputs/<run_dir> [--ckpt last|best] [--log-level INFO]
```

Evaluate a trained run on the test sets of every client (`--save-results` writes `<run_dir>/test_results.csv`):

```shell
uv run python -m scripts.test outputs/<run_dir> [--ckpt best|last] --save-results
```

Monitor training with TensorBoard:

```shell
uv run tensorboard --logdir outputs/
```

## Citation

If you use this code, please cite:

```bibtex
@misc{leclercq2026pafp,
  title         = {Federated Transfer Learning for Medical Image Classification under Mixed Skew},
  author        = {Leclercq, Alexandre G. and Bougleux, S{\'e}bastien and Corroyer-Dulmont, Aur{\'e}lien and H{\'e}rault, Romain},
  year          = {2026},
  eprint        = {2611.05035},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Accepted at FLTA 2026}
}
```

*This entry will be updated with the IEEE proceedings reference once it is published.*
