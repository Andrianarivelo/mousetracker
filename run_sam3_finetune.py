"""Run a generated SAM3 fine-tuning config locally."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from sam3.train.train import single_node_runner
from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to a generated SAM3 fine-tuning YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_omegaconf_resolvers()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Fine-tune config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    cfg.launcher.num_nodes = 1
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = str(config_path.parent)

    experiment_dir = Path(str(cfg.launcher.experiment_log_dir)).resolve()
    makedir(str(experiment_dir))

    print("###################### Fine-Tune Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    with g_pathmgr.open(str(experiment_dir / "config.yaml"), "w") as handle:
        handle.write(OmegaConf.to_yaml(cfg))
    with g_pathmgr.open(str(experiment_dir / "config_resolved.yaml"), "w") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))

    port_range = cfg.get("submitit", {}).get("port_range", [10000, 65000])
    port_low = int(port_range[0])
    port_high = int(port_range[1])
    main_port = random.randint(port_low, port_high)

    print(f"Experiment Log Dir:\n{experiment_dir}")
    single_node_runner(cfg, main_port)


if __name__ == "__main__":
    main()
