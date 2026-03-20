"""Run a generated SAM3 fine-tuning config locally."""

from __future__ import annotations

import argparse
import contextlib
import os
import random
from pathlib import Path

import torch
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from sam3.train.train import single_node_runner
from sam3.train.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def _patch_sam3_single_gpu_windows() -> None:
    """Avoid torch.distributed/DDP for the local 1-GPU Windows training path."""

    if os.name != "nt":
        return

    from sam3.train.data import torch_dataset as torch_dataset_module
    from sam3.train import trainer as trainer_module

    original_barrier = trainer_module.dist.barrier
    original_is_dist_ready = trainer_module.is_dist_avail_and_initialized
    original_torchdataset_init = torch_dataset_module.TorchDataset.__init__
    original_all_reduce = torch.distributed.all_reduce
    original_get_world_size = torch.distributed.get_world_size
    original_get_rank = torch.distributed.get_rank

    def _is_single_process_run() -> bool:
        return int(os.environ.get("WORLD_SIZE", "1")) <= 1

    def _dist_is_ready() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _safe_barrier(*args, **kwargs):
        if _dist_is_ready():
            return original_barrier(*args, **kwargs)
        return None

    def _safe_all_reduce(*args, **kwargs):
        if _is_single_process_run() and not _dist_is_ready():
            return None
        return original_all_reduce(*args, **kwargs)

    def _safe_get_world_size(*args, **kwargs):
        if _is_single_process_run() and not _dist_is_ready():
            return 1
        return original_get_world_size(*args, **kwargs)

    def _safe_get_rank(*args, **kwargs):
        if _is_single_process_run() and not _dist_is_ready():
            return 0
        return original_get_rank(*args, **kwargs)

    def _is_dist_ready_for_trainer() -> bool:
        if _is_single_process_run():
            return True
        return original_is_dist_ready()

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        if _is_single_process_run():
            self.rank = 0
            return

        self.rank = trainer_module.setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        if _is_single_process_run():
            self.model.no_sync = contextlib.nullcontext
            return

        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            static_graph=distributed_conf.static_graph,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = trainer_module.get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
            self.model.register_comm_hook(None, hook)

    def _torchdataset_init(self, *args, **kwargs):
        if _is_single_process_run():
            kwargs["enable_distributed_sampler"] = False
        original_torchdataset_init(self, *args, **kwargs)

    torch.distributed.all_reduce = _safe_all_reduce
    torch.distributed.get_world_size = _safe_get_world_size
    torch.distributed.get_rank = _safe_get_rank
    torch_dataset_module.TorchDataset.__init__ = _torchdataset_init
    trainer_module.dist.barrier = _safe_barrier
    trainer_module.is_dist_avail_and_initialized = _is_dist_ready_for_trainer
    trainer_module.Trainer._setup_torch_dist_and_backend = _setup_torch_dist_and_backend
    trainer_module.Trainer._setup_ddp_distributed_training = (
        _setup_ddp_distributed_training
    )


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
    _patch_sam3_single_gpu_windows()
    print("[MouseTracker] Loading fine-tune configuration...")

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Fine-tune config not found: {config_path}")

    cfg = OmegaConf.load(str(config_path))
    cfg.launcher.num_nodes = 1
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = str(config_path.parent)

    experiment_dir = Path(str(cfg.launcher.experiment_log_dir)).resolve()
    makedir(str(experiment_dir))
    print(f"[MouseTracker] Fine-tune output directory: {experiment_dir}")
    print(f"[MouseTracker] Detailed trainer log: {experiment_dir / 'logs' / 'log.txt'}")

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
    print("[MouseTracker] Initializing trainer and dataloaders...")
    single_node_runner(cfg, main_port)
    print("[MouseTracker] Fine-tune run finished.")


if __name__ == "__main__":
    main()
