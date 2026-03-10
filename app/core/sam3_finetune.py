"""Helpers for building and launching local SAM3 fine-tuning jobs."""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from app.config import ROOT_DIR, get_sam3_checkpoint
from app.core.dataset_manager import ensure_sam3_training_dataset


def default_bpe_path() -> str:
    try:
        asset = resources.files("sam3").joinpath("assets", "bpe_simple_vocab_16e6.txt.gz")
        return str(Path(str(asset)).resolve())
    except Exception:
        return ""


def available_gpu_count() -> int:
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def default_distributed_backend() -> str:
    if platform.system().lower().startswith("win"):
        return "gloo"
    return "nccl"


def default_amp_dtype(accelerator: str) -> str:
    accelerator = str(accelerator).lower().strip()
    if accelerator != "cuda":
        return "float16"
    # Favor fp16 for local Windows training. Single-GPU runs bypass DDP now,
    # but fp16 remains the safer default for this stack.
    if platform.system().lower().startswith("win"):
        return "float16"
    try:
        if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
            if torch.cuda.is_bf16_supported():
                return "bfloat16"
    except Exception:
        pass
    return "float16"


def max_supported_local_gpus() -> int:
    count = max(1, available_gpu_count())
    if default_distributed_backend() == "gloo":
        return 1
    return count


def default_run_name() -> str:
    return "sam3_ft_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_run_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return clean.strip("._-") or default_run_name()


@dataclass
class Sam3FineTuneParams:
    run_name: str
    output_dir: str
    checkpoint_path: str
    bpe_path: str
    max_epochs: int = 20
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 0
    num_gpus: int = 1
    learning_rate_scale: float = 0.1
    resolution: int = 1008
    limit_train_images: int = 0
    val_epoch_freq: int = 10
    gradient_accumulation_steps: int = 1

    def normalized(self) -> "Sam3FineTuneParams":
        return Sam3FineTuneParams(
            run_name=sanitize_run_name(self.run_name),
            output_dir=str(Path(self.output_dir).resolve()),
            checkpoint_path=str(Path(self.checkpoint_path).resolve())
            if self.checkpoint_path
            else "",
            bpe_path=str(Path(self.bpe_path).resolve()) if self.bpe_path else "",
            max_epochs=max(1, int(self.max_epochs)),
            train_batch_size=max(1, int(self.train_batch_size)),
            val_batch_size=max(1, int(self.val_batch_size)),
            num_workers=max(0, int(self.num_workers)),
            num_gpus=min(max(1, int(self.num_gpus)), max_supported_local_gpus()),
            learning_rate_scale=max(float(self.learning_rate_scale), 1e-6),
            resolution=max(256, int(self.resolution)),
            limit_train_images=max(0, int(self.limit_train_images)),
            val_epoch_freq=max(1, int(self.val_epoch_freq)),
            gradient_accumulation_steps=max(1, int(self.gradient_accumulation_steps)),
        )


def default_finetune_params(dataset_dir: str) -> Sam3FineTuneParams:
    base_output = Path(dataset_dir) / "sam3_runs" / default_run_name()
    return Sam3FineTuneParams(
        run_name=base_output.name,
        output_dir=str(base_output),
        checkpoint_path=get_sam3_checkpoint() or "",
        bpe_path=default_bpe_path(),
        num_gpus=max_supported_local_gpus(),
    ).normalized()


def prepare_finetune_job(
    dataset_dir: str,
    params: Sam3FineTuneParams,
    sam3_class_names: Optional[dict[int, str]] = None,
) -> dict:
    """
    Ensure the dataset is SAM3-ready and write a local Hydra config for training.

    Returns:
        {
            "config_path": str,
            "output_dir": str,
            "dataset_info": dict,
        }
    """
    params = params.normalized()
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = ensure_sam3_training_dataset(
        dataset_dir,
        sam3_class_names=sam3_class_names,
    )
    if dataset_info["train_count"] <= 0:
        raise ValueError("The selected dataset has no training images.")

    config = _build_training_config(dataset_info, params)
    config_path = output_dir / "sam3_finetune_config.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(config_path))
    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "dataset_info": dataset_info,
    }


def runner_script_path() -> str:
    return str((ROOT_DIR / "run_sam3_finetune.py").resolve())


def _build_training_config(dataset_info: dict, params: Sam3FineTuneParams) -> dict:
    has_val = int(dataset_info["val_count"]) > 0
    accelerator = "cuda" if available_gpu_count() > 0 else "cpu"
    backend = default_distributed_backend()
    amp_dtype = default_amp_dtype(accelerator)
    enable_distributed_sampler = params.num_gpus > 1
    checkpoint_path = params.checkpoint_path or None
    bpe_path = params.bpe_path or default_bpe_path()

    train_dataset = {
        "_target_": "sam3.train.data.torch_dataset.TorchDataset",
        "dataset": {
            "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
            "limit_ids": params.limit_train_images or None,
            "transforms": "${ft.train_transforms}",
            "load_segmentation": True,
            "max_ann_per_img": 500000,
            "multiplier": 1,
            "max_train_queries": 50000,
            "max_val_queries": 50000,
            "training": True,
            "use_caching": False,
            "img_folder": dataset_info["train_images_dir"],
            "ann_file": dataset_info["train_annotations"],
        },
        "shuffle": True,
        "batch_size": params.train_batch_size,
        "num_workers": params.num_workers,
        "pin_memory": accelerator == "cuda",
        "drop_last": True,
        "collate_fn": "${scratch.collate_fn}",
        "enable_distributed_sampler": enable_distributed_sampler,
    }

    val_dataset = None
    if has_val:
        val_dataset = {
            "_target_": "sam3.train.data.torch_dataset.TorchDataset",
            "dataset": {
                "_target_": "sam3.train.data.sam3_image_dataset.Sam3ImageDataset",
                "load_segmentation": True,
                "coco_json_loader": {
                    "_target_": "sam3.train.data.coco_json_loaders.COCO_FROM_JSON",
                    "include_negatives": True,
                    "category_chunk_size": 2,
                    "_partial_": True,
                },
                "img_folder": dataset_info["val_images_dir"],
                "ann_file": dataset_info["val_annotations"],
                "transforms": "${ft.val_transforms}",
                "max_ann_per_img": 100000,
                "multiplier": 1,
                "training": False,
            },
            "shuffle": False,
            "batch_size": params.val_batch_size,
            "num_workers": 0,
            "pin_memory": accelerator == "cuda",
            "drop_last": False,
            "collate_fn": "${scratch.collate_fn_val}",
            "enable_distributed_sampler": enable_distributed_sampler,
        }

    return {
        "defaults": ["_self_"],
        "paths": {
            "dataset_root": str(Path(dataset_info["train_images_dir"]).parent.parent),
            "experiment_log_dir": params.output_dir,
            "bpe_path": bpe_path,
            "checkpoint_path": checkpoint_path,
        },
        "ft": {
            "train_transforms": [
                {
                    "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
                    "transforms": [
                        {
                            "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
                            "query_filter": {
                                "_target_": "sam3.train.transforms.filter_query_transforms.FilterCrowds"
                            },
                        },
                        {
                            "_target_": "sam3.train.transforms.point_sampling.RandomizeInputBbox",
                            "box_noise_std": 0.1,
                            "box_noise_max": 20,
                        },
                        {"_target_": "sam3.train.transforms.segmentation.DecodeRle"},
                        {
                            "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                            "sizes": {
                                "_target_": "sam3.train.transforms.basic.get_random_resize_scales",
                                "size": "${scratch.resolution}",
                                "min_size": 480,
                                "rounded": False,
                            },
                            "max_size": {
                                "_target_": "sam3.train.transforms.basic.get_random_resize_max_size",
                                "size": "${scratch.resolution}",
                            },
                            "square": True,
                            "consistent_transform": False,
                        },
                        {
                            "_target_": "sam3.train.transforms.basic_for_api.PadToSizeAPI",
                            "size": "${scratch.resolution}",
                            "consistent_transform": False,
                        },
                        {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
                        {
                            "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
                            "query_filter": {
                                "_target_": "sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"
                            },
                        },
                        {
                            "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                            "mean": "${scratch.train_norm_mean}",
                            "std": "${scratch.train_norm_std}",
                        },
                        {
                            "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
                            "query_filter": {
                                "_target_": "sam3.train.transforms.filter_query_transforms.FilterEmptyTargets"
                            },
                        },
                    ],
                },
                {
                    "_target_": "sam3.train.transforms.filter_query_transforms.FlexibleFilterFindGetQueries",
                    "query_filter": {
                        "_target_": "sam3.train.transforms.filter_query_transforms.FilterFindQueriesWithTooManyOut",
                        "max_num_objects": "${scratch.max_ann_per_img}",
                    },
                },
            ],
            "val_transforms": [
                {
                    "_target_": "sam3.train.transforms.basic_for_api.ComposeAPI",
                    "transforms": [
                        {
                            "_target_": "sam3.train.transforms.basic_for_api.RandomResizeAPI",
                            "sizes": "${scratch.resolution}",
                            "max_size": {
                                "_target_": "sam3.train.transforms.basic.get_random_resize_max_size",
                                "size": "${scratch.resolution}",
                            },
                            "square": True,
                            "consistent_transform": False,
                        },
                        {"_target_": "sam3.train.transforms.basic_for_api.ToTensorAPI"},
                        {
                            "_target_": "sam3.train.transforms.basic_for_api.NormalizeAPI",
                            "mean": "${scratch.train_norm_mean}",
                            "std": "${scratch.train_norm_std}",
                        },
                    ],
                }
            ],
            "loss": {
                "_target_": "sam3.train.loss.sam3_loss.Sam3LossWrapper",
                "matcher": "${scratch.matcher}",
                "o2m_weight": 2.0,
                "o2m_matcher": {
                    "_target_": "sam3.train.matcher.BinaryOneToManyMatcher",
                    "alpha": 0.3,
                    "threshold": 0.4,
                    "topk": 4,
                },
                "use_o2m_matcher_on_o2m_aux": False,
                "loss_fns_find": [
                    {
                        "_target_": "sam3.train.loss.loss_fns.Boxes",
                        "weight_dict": {
                            "loss_bbox": 5.0,
                            "loss_giou": 2.0,
                        },
                    },
                    {
                        "_target_": "sam3.train.loss.loss_fns.IABCEMdetr",
                        "weak_loss": False,
                        "weight_dict": {
                            "loss_ce": 20.0,
                            "presence_loss": 20.0,
                        },
                        "pos_weight": 10.0,
                        "alpha": 0.25,
                        "gamma": 2,
                        "use_presence": True,
                        "pos_focal": False,
                        "pad_n_queries": 200,
                        "pad_scale_pos": 1.0,
                    },
                    {
                        "_target_": "sam3.train.loss.loss_fns.Masks",
                        "focal_alpha": 0.25,
                        "focal_gamma": 2.0,
                        "weight_dict": {
                            "loss_mask": 200.0,
                            "loss_dice": 10.0,
                        },
                        "compute_aux": False,
                    },
                ],
                "loss_fn_semantic_seg": None,
                "scale_by_find_batch_size": True,
            },
        },
        "scratch": {
            "enable_segmentation": True,
            "resolution": params.resolution,
            "max_ann_per_img": 200,
            "train_norm_mean": [0.5, 0.5, 0.5],
            "train_norm_std": [0.5, 0.5, 0.5],
            "val_norm_mean": [0.5, 0.5, 0.5],
            "val_norm_std": [0.5, 0.5, 0.5],
            "lr_scale": params.learning_rate_scale,
            "lr_transformer": "${times:8e-4,${scratch.lr_scale}}",
            "lr_vision_backbone": "${times:2.5e-4,${scratch.lr_scale}}",
            "lr_language_backbone": "${times:5e-5,${scratch.lr_scale}}",
            "lrd_vision_backbone": 0.9,
            "wd": 0.1,
            "scheduler_timescale": 20,
            "scheduler_warmup": 20,
            "scheduler_cooldown": 20,
            "hybrid_repeats": 1,
            "collate_fn": {
                "_target_": "sam3.train.data.collator.collate_fn_api",
                "_partial_": True,
                "repeats": "${scratch.hybrid_repeats}",
                "dict_key": "all",
                "with_seg_masks": True,
            },
            "collate_fn_val": {
                "_target_": "sam3.train.data.collator.collate_fn_api",
                "_partial_": True,
                "repeats": "${scratch.hybrid_repeats}",
                "dict_key": "val",
                "with_seg_masks": True,
            },
            "matcher": {
                "_target_": "sam3.train.matcher.BinaryHungarianMatcherV2",
                "focal": True,
                "cost_class": 2.0,
                "cost_bbox": 5.0,
                "cost_giou": 2.0,
                "alpha": 0.25,
                "gamma": 2,
                "stable": False,
            },
        },
        "trainer": {
            "_target_": "sam3.train.trainer.Trainer",
            "skip_saving_ckpts": False,
            "empty_gpu_mem_cache_after_eval": True,
            "skip_first_val": True,
            "max_epochs": params.max_epochs,
            "accelerator": accelerator,
            "seed_value": 123,
            "val_epoch_freq": params.val_epoch_freq,
            "mode": "train" if has_val else "train_only",
            "gradient_accumulation_steps": params.gradient_accumulation_steps,
            "distributed": {
                "backend": backend,
                "find_unused_parameters": True,
                "gradient_as_bucket_view": True,
            },
            "loss": {
                "all": "${ft.loss}",
                "default": {
                    "_target_": "sam3.train.loss.sam3_loss.DummyLoss",
                    "device": accelerator,
                },
            },
            "data": {
                "train": train_dataset,
                "val": val_dataset,
            },
            "model": {
                "_target_": "sam3.model_builder.build_sam3_image_model",
                "bpe_path": "${paths.bpe_path}",
                "device": "cpu",
                "eval_mode": False,
                "enable_segmentation": True,
                "checkpoint_path": "${paths.checkpoint_path}",
            },
            "meters": {},
            "optim": {
                "amp": {
                    "enabled": accelerator == "cuda",
                    "amp_dtype": amp_dtype,
                },
                "optimizer": {"_target_": "torch.optim.AdamW"},
                "gradient_clip": {
                    "_target_": "sam3.train.optim.optimizer.GradientClipper",
                    "max_norm": 0.1,
                    "norm_type": 2,
                },
                "param_group_modifiers": [
                    {
                        "_target_": "sam3.train.optim.optimizer.layer_decay_param_modifier",
                        "_partial_": True,
                        "layer_decay_value": "${scratch.lrd_vision_backbone}",
                        "apply_to": "backbone.vision_backbone.trunk",
                        "overrides": [
                            {"pattern": "*pos_embed*", "value": 1.0},
                        ],
                    }
                ],
                "options": {
                    "lr": [
                        {
                            "scheduler": {
                                "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                                "base_lr": "${scratch.lr_transformer}",
                                "timescale": "${scratch.scheduler_timescale}",
                                "warmup_steps": "${scratch.scheduler_warmup}",
                                "cooldown_steps": "${scratch.scheduler_cooldown}",
                            }
                        },
                        {
                            "scheduler": {
                                "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                                "base_lr": "${scratch.lr_vision_backbone}",
                                "timescale": "${scratch.scheduler_timescale}",
                                "warmup_steps": "${scratch.scheduler_warmup}",
                                "cooldown_steps": "${scratch.scheduler_cooldown}",
                            },
                            "param_names": ["backbone.vision_backbone.*"],
                        },
                        {
                            "scheduler": {
                                "_target_": "sam3.train.optim.schedulers.InverseSquareRootParamScheduler",
                                "base_lr": "${scratch.lr_language_backbone}",
                                "timescale": "${scratch.scheduler_timescale}",
                                "warmup_steps": "${scratch.scheduler_warmup}",
                                "cooldown_steps": "${scratch.scheduler_cooldown}",
                            },
                            "param_names": ["backbone.language_backbone.*"],
                        },
                    ],
                    "weight_decay": [
                        {
                            "scheduler": {
                                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                "value": "${scratch.wd}",
                            }
                        },
                        {
                            "scheduler": {
                                "_target_": "fvcore.common.param_scheduler.ConstantParamScheduler",
                                "value": 0.0,
                            },
                            "param_names": ["*bias*"],
                            "module_cls_names": ["torch.nn.LayerNorm"],
                        },
                    ],
                },
            },
            "checkpoint": {
                "save_dir": "${launcher.experiment_log_dir}/checkpoints",
                "save_freq": 0,
            },
            "logging": {
                "tensorboard_writer": {
                    "_target_": "sam3.train.utils.logger.make_tensorboard_logger",
                    "log_dir": "${launcher.experiment_log_dir}/tensorboard",
                    "flush_secs": 120,
                    "should_log": True,
                },
                "wandb_writer": None,
                "log_dir": "${launcher.experiment_log_dir}/logs",
                "log_freq": 10,
            },
        },
        "launcher": {
            "num_nodes": 1,
            "gpus_per_node": min(params.num_gpus, max(1, available_gpu_count() or 1)),
            "experiment_log_dir": params.output_dir,
            "multiprocessing_context": "spawn",
        },
        "submitit": {
            "account": None,
            "partition": None,
            "qos": None,
            "timeout_hour": 72,
            "use_cluster": False,
            "cpus_per_task": max(1, params.num_workers or 1),
            "port_range": [10000, 65000],
            "constraint": None,
        },
    }
