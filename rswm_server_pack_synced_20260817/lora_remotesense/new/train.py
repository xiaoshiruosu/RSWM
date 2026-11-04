#!/usr/bin/env python3
"""
LingBot-World 遥感 LoRA 微调训练脚本 — Ampere 80GB 适配版

基于 lora_remotesense/train.py，针对单卡 A800/A100 80GB 做工程优化（不改实验任务）：
- A800 profile：VAE/T5 to CPU + GC + bf16 + 单专家 GPU 驻留
- inactive_model_cpu 会连同 LoRA 梯度与 torch.AdamW 状态一起换入/换出，支持梯度累积
- CPU offload 禁用 bitsandbytes AdamW8bit（bnb 不支持这种混合 CPU/GPU 参数生命周期）
- compute_loss 内主动释放 VAE 中间张量 + empty_cache
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 默认开

显存预算（A800/A100 80GB, B=1）：
  1× 14B DiT (bf16, on GPU)        ~28 GB
  LoRA grads + torch AdamW state    依 rank/targets 而变
  Activations (GC on, T=33/81)      依帧数而变
  Temp buffers (VAE/T5 swap)        ~3-5 GB

实验关键超参 (lora_rank, lora_alpha, max_frames) 保持与原版一致，仅工程层面省显存。
"""

import os
import sys
import argparse
import logging
import time
import json
import signal
import traceback
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW as TorchAdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import get_peft_model, PeftModel, set_peft_model_state_dict
from tqdm import tqdm
import numpy as np
from einops import rearrange

# new/ 自身在 sys.path 上（让 dataset_loader.py 等本地副本可被 import）
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# wan/ 解析：优先用 new/wan/（打包上传场景），fallback 上溯 lingbot/wan/（开发场景）
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.animate.animate_utils import get_loraconfig
from wan.utils.cam_utils import (
    interpolate_camera_poses,
    compute_relative_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from dataset_loader import (LingBotDataset, split_dataset,
                            split_dataset_from_file)  # new/ 本地副本
from wan.distributed.util import init_distributed_group


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_interrupt_requested = False


def _is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _signal_handler(signum, frame):
    global _interrupt_requested
    _interrupt_requested = True
    logger.info("收到中断信号，将在当前 epoch 完成后保存并退出...")


signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _signal_handler)


# --- Parameter-efficiency audit (advisor point 8: prove LoRA params are small) ---
# Module-class buckets for the trainable-param breakdown. Substrings mirror
# wan/modules/animate/animate_utils.py:16-17 (_ATTN_SUBSTRINGS / _CAM_SUBSTRINGS) so the
# audit classes line up 1:1 with what --lora_targets {all,attn_cam,attn_only} selects.
_AUDIT_CLASSES = (
    ("self_attn", ("self_attn",)),
    ("cross_attn", ("cross_attn",)),
    ("cam", ("cam_injector", "cam_scale", "cam_shift")),
    ("ffn", ("ffn",)),
)
_AUDIT_CLASS_NAMES = tuple(c for c, _ in _AUDIT_CLASSES) + ("other",)


def _classify_param_name(name: str) -> str:
    """Map a parameter name to one module class (first match wins; else 'other')."""
    for cls, subs in _AUDIT_CLASSES:
        if any(s in name for s in subs):
            return cls
    return "other"


def _audit_trainable_params(model, expert_name: str, args) -> dict:
    """One pass over named_parameters(): total/trainable/ratio + per-module-class breakdown.

    Fail-loud contract: any trainable parameter whose name does NOT contain 'lora' means the
    frozen backbone leaked into the optimizer -> RuntimeError (never warn-and-continue).
    Returns a small JSON-serializable dict; writes no files itself.
    """
    total = 0
    trainable = 0
    by_class = {
        c: {"n_tensors": 0, "n_params": 0} for c in _AUDIT_CLASS_NAMES
    }
    non_lora_trainable: List[str] = []

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if not p.requires_grad:
            continue
        trainable += n
        cls = _classify_param_name(name)
        by_class[cls]["n_tensors"] += 1
        by_class[cls]["n_params"] += n
        if "lora" not in name.lower():
            non_lora_trainable.append(name)

    audit = {
        "expert": expert_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": args.lora_targets,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_ratio_pct": (100.0 * trainable / total) if total else 0.0,
        "trainable_by_module_class": {
            c: dict(v, pct_of_trainable=(100.0 * v["n_params"] / trainable) if trainable else 0.0)
            for c, v in by_class.items()
        },
        "non_lora_trainable_params": non_lora_trainable,
    }

    if _is_main_process():
        logger.info(
            f"[param-audit:{expert_name}] targets={args.lora_targets} "
            f"rank={args.lora_rank} alpha={args.lora_alpha}"
        )
        logger.info(
            f"[param-audit:{expert_name}] trainable {trainable:,} / total {total:,} "
            f"({audit['trainable_ratio_pct']:.4f}%)"
        )
        for c in _AUDIT_CLASS_NAMES:
            v = audit["trainable_by_module_class"][c]
            if v["n_tensors"] == 0:
                continue
            logger.info(
                f"[param-audit:{expert_name}]   {c:<11s} "
                f"tensors={v['n_tensors']:<5d} params={v['n_params']:>12,} "
                f"({v['pct_of_trainable']:.2f}% of trainable)"
            )

    # Fail-loud: backbone must stay frozen. Raise on EVERY rank (not just main) so a DDP
    # run aborts instead of deadlocking on the next collective.
    if non_lora_trainable:
        shown = non_lora_trainable[:20]
        more = len(non_lora_trainable) - len(shown)
        raise RuntimeError(
            f"[param-audit:{expert_name}] BACKBONE PARAMS ARE UNEXPECTEDLY UNFROZEN: "
            f"{len(non_lora_trainable)} trainable parameter(s) have no 'lora' in their name, "
            f"so the frozen {expert_name}-noise backbone would be updated and the "
            f"parameter-efficiency claim would be invalid. Offending names: {shown}"
            + (f" ... (+{more} more)" if more else "")
        )

    return audit


def _build_optimizer(params, lr: float, weight_decay: float, use_8bit: bool):
    """构建优化器：优先 bnb 8bit AdamW；失败回退 torch AdamW。"""
    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
            if _is_main_process():
                logger.info("已启用 bitsandbytes.AdamW8bit (LoRA optimizer state = 8bit)")
            return opt
        except Exception as e:
            if _is_main_process():
                logger.warning(
                    f"bitsandbytes 不可用，回退 torch.AdamW: {repr(e)}"
                )
    if _is_main_process():
        logger.info("使用 torch.AdamW (optimizer state = fp32)")
    return TorchAdamW(params, lr=lr, weight_decay=weight_decay)


class RemoteSenseLoRATrainer:
    """A800/A100 80GB 与 H200 双 profile 遥感 LoRA 训练器。"""

    def __init__(self, args):
        self.args = args
        self.use_ddp = _is_distributed()
        if self.use_ddp:
            self.local_rank = _get_local_rank()
        elif torch.cuda.is_available():
            if args.device_id is not None:
                torch.cuda.set_device(args.device_id)
            self.local_rank = torch.cuda.current_device()
        else:
            self.local_rank = 0
        self.world_size = _get_world_size()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if self.use_ddp and torch.cuda.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = WAN_CONFIGS["i2v-A14B"]
        self.boundary_step = int(self.config.boundary * self.config.num_train_timesteps)
        self._data_prompt_schema = None

        # [v2.8.2 C3] pose_mode↔H5 一致性在【加载任何模型之前】就校验。
        # 旧位置在第一个 training step 里(compute_loss 惰性检查),于是每次误配
        # 都要先把两个 14B DiT + VAE + T5 全加载完、跑完第一批 VAE encode 才炸,
        # 白烧十几分钟 GPU 分配 —— 而误配恰恰是最常见的失败(见 C3 那批老入口)。
        self._assert_pose_mode(getattr(args, "pose_mode", "native"))
        self._assert_prompt_schema()
        self._pose_mode_checked = True
        # [v2.9.1] 续训门同样提前:它只读一个 JSON,却曾排在两个 14B 加载 +
        # LoRA 权重载入之后 —— 与刚做的"省掉误配那十几分钟"是同一件事。
        if args.resume_from:
            self._assert_resume_pose_mode(args.resume_from)

        # A800 profile 依赖单专家 CPU/GPU swap，而经典 DDP 要求构造时所有参数
        # 已在对应 GPU。两者不能混用；在加载 28B 权重前立即失败，避免占用共享卡。
        if self.use_ddp and args.inactive_model_cpu:
            raise RuntimeError(
                "当前 A800 single-expert offload 路径不兼容 DDP。单卡 smoke/正式训练"
                "请只暴露一张卡；多卡必须显式选择不 offload 的 H200 profile，或等待"
                "经验证的 FSDP/model-parallel 实现。未启动任何训练。")

        self._init_vae()
        self._init_text_encoder()
        self._init_models()

        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.training_log: List[dict] = []

        if args.resume_from:
            self._load_resume_lora_weights(args.resume_from)

        if self.use_ddp:
            self._wrap_ddp()

        self._init_dataset()
        self._init_optimizer()

        if args.resume_from:
            self._load_resume_training_state(args.resume_from)

        self.grad_scaler = None
        if self.args.mixed_precision == "fp16":
            self.grad_scaler = torch.amp.GradScaler("cuda")
            if _is_main_process():
                logger.info("已启用 fp16 GradScaler")

        if self.args.inactive_model_cpu and _is_main_process():
            logger.info(
                "已启用 --inactive_model_cpu：每个 step 仅当前 timestep 子集对应模型驻 GPU，"
                "另一份 swap 至 CPU（节省约 28GB，速度有损）"
            )

        logger.info(
            f"训练器初始化完成，设备: {self.device}"
            + (f" (DDP world_size={self.world_size})" if self.use_ddp else "")
        )

    # ------------------------------------------------------------------ init
    def _init_vae(self):
        if _is_main_process():
            logger.info("加载 VAE...")
        vae_path = os.path.join(self.args.checkpoint_dir, self.config.vae_checkpoint)
        vae_device = "cpu" if self.args.vae_cpu else self.device
        self.vae = Wan2_1_VAE(vae_pth=vae_path, device=vae_device)
        self.vae.model.eval().requires_grad_(False)
        if _is_main_process():
            logger.info("VAE 加载完成" + ("（vae_cpu 模式）" if self.args.vae_cpu else ""))

    def _vae_encode(self, videos):
        if self.args.vae_cpu:
            self.vae.model.to(self.device)
            self.vae.mean = self.vae.mean.to(self.device)
            self.vae.std = self.vae.std.to(self.device)
            self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]
        out = self.vae.encode(videos)
        if self.args.vae_cpu:
            self.vae.model.cpu()
            self.vae.mean = self.vae.mean.cpu()
            self.vae.std = self.vae.std.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return out

    def _init_text_encoder(self):
        if _is_main_process():
            logger.info("加载 T5 编码器...")
        t5_device = "cpu" if self.args.t5_cpu else self.device
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(
                self.args.checkpoint_dir, self.config.t5_checkpoint
            ),
            tokenizer_path=os.path.join(
                self.args.checkpoint_dir, self.config.t5_tokenizer
            ),
        )
        self.text_encoder.model.eval().requires_grad_(False)
        if _is_main_process():
            logger.info("T5 编码器加载完成")

    def _init_models(self):
        if _is_main_process():
            logger.info("加载 DiT 模型并配置 LoRA...")

        self.low_noise_model = WanModel.from_pretrained(
            self.args.checkpoint_dir,
            subfolder=self.config.low_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        lora_cfg_low = get_loraconfig(
            self.low_noise_model,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            init_lora_weights="gaussian",
            targets=self.args.lora_targets,
        )
        self.low_noise_model = get_peft_model(self.low_noise_model, lora_cfg_low)

        self.high_noise_model = WanModel.from_pretrained(
            self.args.checkpoint_dir,
            subfolder=self.config.high_noise_checkpoint,
            torch_dtype=torch.bfloat16,
        )
        lora_cfg_high = get_loraconfig(
            self.high_noise_model,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            init_lora_weights="gaussian",
            targets=self.args.lora_targets,
        )
        self.high_noise_model = get_peft_model(self.high_noise_model, lora_cfg_high)

        if self.args.inactive_model_cpu:
            self.low_noise_model.to(dtype=torch.bfloat16)
            self.high_noise_model.to(dtype=torch.bfloat16)
            self.low_noise_model.cpu()
            self.high_noise_model.cpu()
            self._model_on_gpu = {"low": False, "high": False}
        else:
            self.low_noise_model.to(device=self.device, dtype=torch.bfloat16)
            self.high_noise_model.to(device=self.device, dtype=torch.bfloat16)
            self._model_on_gpu = {"low": True, "high": True}

        if self.args.gradient_checkpointing:
            self.low_noise_model.gradient_checkpointing_enable()
            self.high_noise_model.gradient_checkpointing_enable()
            if _is_main_process():
                logger.info("已启用梯度检查点 (gradient checkpointing)")

        # 参数效率审计 (advisor point 8)。在两个专家都 get_peft_model() 之后执行:
        # 每专家一遍 named_parameters() -> 总量/可训练/占比 + 按模块类拆分 + 骨干解冻检查
        # (非 lora 可训练参数 -> 立即 RuntimeError)。审计结果落一个 param_audit.json。
        audit_low = _audit_trainable_params(self.low_noise_model, "low", self.args)
        audit_high = _audit_trainable_params(self.high_noise_model, "high", self.args)

        total_trainable = audit_low["trainable_params"] + audit_high["trainable_params"]
        total_all = audit_low["total_params"] + audit_high["total_params"]
        self.param_audit = {
            "schema_version": 1,
            "generated_by": "lora_remotesense/new/train.py:_audit_trainable_params",
            "timestamp": datetime.now().isoformat(),
            "lora_rank": self.args.lora_rank,
            "lora_alpha": self.args.lora_alpha,
            "lora_targets": self.args.lora_targets,
            "experts": {"low": audit_low, "high": audit_high},
            # combined = 实际部署的适配器 (low+high 两个专家都要存/加载),
            # 论文里报参数效率应以此为准。
            "combined": {
                "trainable_params": total_trainable,
                "total_params": total_all,
                "trainable_ratio_pct": (
                    100.0 * total_trainable / total_all if total_all else 0.0
                ),
            },
        }

        if _is_main_process():
            logger.info(
                f"[param-audit:combined] low+high 可训练 {total_trainable:,} / "
                f"总计 {total_all:,} "
                f"({self.param_audit['combined']['trainable_ratio_pct']:.4f}%)"
            )
            # 单文件写出(inode 紧张: 绝不按参数写小文件)
            out_dir = Path(self.args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            audit_path = out_dir / "param_audit.json"
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(self.param_audit, f, indent=2, ensure_ascii=False)
            logger.info(f"参数效率审计已保存: {audit_path}")

    def _ensure_on_gpu(self, which: str):
        if not self.args.inactive_model_cpu:
            return
        if which == "low" and not self._model_on_gpu["low"]:
            self.low_noise_model.to(self.device, non_blocking=True)
            self._move_optimizer_state(self.low_noise_model, self.device)
            self._model_on_gpu["low"] = True
        if which == "high" and not self._model_on_gpu["high"]:
            self.high_noise_model.to(self.device, non_blocking=True)
            self._move_optimizer_state(self.high_noise_model, self.device)
            self._model_on_gpu["high"] = True

    def _offload_to_cpu(self, which: str):
        if not self.args.inactive_model_cpu:
            return
        if which == "low" and self._model_on_gpu["low"]:
            self.low_noise_model.cpu()
            self._move_optimizer_state(self.low_noise_model, torch.device("cpu"))
            self._model_on_gpu["low"] = False
        if which == "high" and self._model_on_gpu["high"]:
            self.high_noise_model.cpu()
            self._move_optimizer_state(self.high_noise_model, torch.device("cpu"))
            self._model_on_gpu["high"] = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _move_optimizer_state(self, model, device) -> None:
        """让 LoRA optimizer state 与参数同设备。

        ``Module.cpu()/to()`` 会移动 Parameter 及其 grad，却不会移动 optimizer.state。
        A800 的双专家交换若漏掉这一步，AdamW 下一次更新会出现 CPU/CUDA device
        mismatch。这里只移动可训练 LoRA 参数的少量状态，不复制 14B 冻结骨干状态。
        """
        optimizer = getattr(self, "optimizer", None)
        if optimizer is None:
            return
        for param in model.parameters():
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device, non_blocking=True)

    def _offload_all_experts(self) -> None:
        if not self.args.inactive_model_cpu:
            return
        self._offload_to_cpu("low")
        self._offload_to_cpu("high")

    def _init_dataset(self):
        if _is_main_process():
            logger.info(f"加载数据集: {self.args.data_path}")
        full_dataset = LingBotDataset(
            h5_path=self.args.data_path,
            normalize=True,
            max_frames=self.args.max_frames,
        )
        # 划分种子与训练种子解耦:多训练种子实验必须固定 split_seed(=42),
        # 否则每个种子训到不同划分 → 共享验证集被泄漏进训练、种子间不可比。
        split_seed = self.args.split_seed if self.args.split_seed is not None else self.args.seed
        if self.args.split_file:
            train_sub, val_sub, train_idx, val_idx = split_dataset_from_file(
                full_dataset, self.args.split_file)
        else:
            train_sub, val_sub, train_idx, val_idx = split_dataset(
                full_dataset,
                train_ratio=self.args.train_ratio,
                val_ratio=self.args.val_ratio,
                seed=split_seed,
            )
        self.train_dataset = train_sub
        self.val_dataset = val_sub
        self.train_indices = train_idx
        self.val_indices = val_idx

        # [RSRoll-Bench P0#9] 训练样本硬上限（成本闸门）。
        # 为什么需要：此前没有任何样本数上限，一次配置错误就能把 4000 样本 × 5 epoch
        # 排进队列；它的失败方式不是崩溃，而是 walltime 被杀 + 自动 --resume_from
        # 继续跑 —— 看起来一直在推进，实际把整个投稿窗口吃掉。
        # 截断是**确定性**的（取划分后的前 N 个，划分本身已由 split_seed 定序），
        # 因此同一 (split_seed, N) 永远得到同一子集，可复现、可写进 manifest。
        _cap = getattr(self.args, "max_train_samples", None)
        if _cap is not None and _cap > 0 and _cap < len(self.train_dataset):
            from torch.utils.data import Subset as _Subset
            _keep = list(range(int(_cap)))
            self.train_dataset = _Subset(self.train_dataset, _keep)
            self.train_indices = [self.train_indices[i] for i in _keep]
            if _is_main_process():
                logger.warning(
                    "[P0#9] 训练集被 --max_train_samples=%d 截断：%d → %d 个样本。"
                    "这是成本闸门，不是数据缺失；manifest 会记录该值。",
                    _cap, len(train_sub), len(self.train_dataset))

        # Real-model smoke must also cap validation.  Otherwise a one-sample
        # training cap can still evaluate the complete official validation set
        # and turn a safety smoke into a many-hour A800 job.
        _val_cap = getattr(self.args, "max_val_samples", None)
        if (_val_cap is not None and _val_cap > 0
                and _val_cap < len(self.val_dataset)):
            from torch.utils.data import Subset as _Subset
            _keep = list(range(int(_val_cap)))
            self.val_dataset = _Subset(self.val_dataset, _keep)
            self.val_indices = [self.val_indices[i] for i in _keep]
            if _is_main_process():
                logger.warning(
                    "验证集被 --max_val_samples=%d 截断：%d → %d 个样本。"
                    "该参数仅用于 smoke/成本控制，manifest 会记录。",
                    _val_cap, len(val_sub), len(self.val_dataset))

        if len(self.train_dataset) == 0:
            raise ValueError("训练集为空，请检查数据路径和划分比例。")

        train_sampler = None
        if self.use_ddp:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.train_sampler = train_sampler

        pin_mem = bool(getattr(self.args, "pin_memory", False))
        train_loader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            collate_fn=self._collate_fn,
        )
        if self.args.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = self.args.prefetch_factor
            train_loader_kwargs["persistent_workers"] = self.args.persistent_workers
        self.train_loader = DataLoader(**train_loader_kwargs)
        val_sampler = None
        if self.use_ddp and len(self.val_dataset) > 0:
            val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        val_loader_kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=pin_mem,
            drop_last=False,
            collate_fn=self._collate_fn,
        )
        if self.args.num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = self.args.prefetch_factor
            val_loader_kwargs["persistent_workers"] = self.args.persistent_workers
        self.val_loader = DataLoader(**val_loader_kwargs)
        if _is_main_process():
            logger.info(
                f"训练样本: {len(self.train_dataset)}, "
                f"验证样本: {len(self.val_dataset)}"
            )

    def _collate_fn(self, batch):
        # [v2.8.1 P1] 非 4n+1 与批内混长度,一律硬错,不再静默截断。
        #
        # 静默截断是本轮 P0 的同族病因:它悄悄把样本的时间轴改短,而 prompt 的
        # 净方向句、waypoints 真值、语义门判定全是按【完整】轨迹算的 —— 截断后
        # 视频只走了前半程,目标却还是全程的净方向,模型被要求学一个数据里
        # 根本没有的对应关系,而症状是完全静默的(loss 照降、数字照出)。
        lens = sorted({int(b["video_frames"].shape[0]) for b in batch})
        raw_min = lens[0]
        if len(lens) > 1:
            raise ValueError(
                f"批内帧数不一致 {lens} —— collate 会把长样本截到 {raw_min} 帧,"
                f"于是 K=4 的样本被当 K=2 训练,而 prompt/waypoints 真值仍是全程"
                f"净方向,prompt 与视频不符。请用 batch_size=1(canonical),或"
                f"实现按长度分桶的 batch_sampler。")
        if (raw_min - 1) % 4 != 0:
            raise ValueError(
                f"帧数 {raw_min} 不满足 4n+1(VAE 时间下采样 4×)。旧版会静默截到 "
                f"{((raw_min - 1) // 4) * 4 + 1} 帧丢掉真实终点。canonical 建库请用 "
                f"--steps 9 → T=17/25/33;单段探针 K=1 → T=9。")
        min_frames = raw_min
        if min_frames < 5:
            raise ValueError(
                f"数据集帧数过少: {min_frames}。每个样本至少需要 5 帧(4n+1, n>=1)。")

        images = torch.stack([b["image"] for b in batch])
        prompts = [b["prompt"] for b in batch]
        video_frames = torch.stack([b["video_frames"][:min_frames] for b in batch])

        intrinsics_list = []
        poses_list = []
        for b in batch:
            inv = b["intrinsics"][:min_frames]
            pos = b["poses"][:min_frames]
            if inv.shape[0] < min_frames:
                pad = inv[-1:].expand(min_frames - inv.shape[0], -1)
                inv = torch.cat([inv, pad], dim=0)
            if pos.shape[0] < min_frames:
                pad = pos[-1:].repeat(min_frames - pos.shape[0], 1, 1)
                pos = torch.cat([pos, pad], dim=0)
            intrinsics_list.append(inv)
            poses_list.append(pos)

        return {
            "image": images,
            "video_frames": video_frames,
            "intrinsics": torch.stack(intrinsics_list),
            "poses": torch.stack(poses_list),
            "prompt": prompts,
            "num_frames": min_frames,
        }

    # -------------------------------------------------------------- plucker
    def _assert_pose_mode(self, pose_mode: str) -> None:
        """[v2.8 ④] pose_mode 与数据集 H5 的 pose_injection 必须自洽,冲突硬错。

        约定:
          * native   ↔ H5 pose_injection = plucker_from_trajectory/waypoints
          * identity ↔ H5 pose_injection = identity_neutralized(静止相机臂)
          * drop     ↔ 任何 H5 都可(位姿根本不传),但用中和过的 H5 属浪费,
                       仍放行并记录 —— 它与 identity 的区别正是本轮要测的。
        误配会让"三个臂只差一个变量"的对比失效,而症状是静默的(数字照出)。
        """
        h5 = getattr(self.args, "data_path", None)
        inj = None
        sample_identity = None
        if h5:
            try:
                import h5py
                import numpy as _np
                with h5py.File(h5, "r") as f:
                    if "metadata" in f and "pose_injection" in f["metadata"].attrs:
                        _pi = f["metadata"].attrs["pose_injection"]
                        # [v2.8.2 N4] 与 evaluate.py 同款解码。h5py 3.x 对字符串
                        # attr 返回 str,但定长 bytes attr 下 str(b'...') 会变成
                        # "b'identity_neutralized'" —— 子串判定侥幸仍对,日志撒谎。
                        inj = _pi.decode() if isinstance(_pi, bytes) else str(_pi)
                    # [v2.8.2 N3] 不能只信 attr:GEO_H5 根本没有 pose_injection,
                    # 于是 inj is None → 旧代码直接 return,几何臂的门形同虚设
                    # ("把中和过的 H5 改名成 train_geo.h5"这类事故抓不到)。
                    # 兜底看第一个样本的真实位姿矩阵 —— 数据级真源。
                    if "data" in f:
                        _ks = list(f["data"].keys())
                        if _ks and "poses" in f["data"][_ks[0]]:
                            _po = _np.asarray(f["data"][_ks[0]]["poses"][:],
                                              dtype=_np.float64)
                            sample_identity = bool(
                                _np.allclose(_po, _np.eye(4), atol=1e-6))
            except Exception as e:
                # [v2.8.2 N3] fail-closed:pose 口径是三臂对比的唯一区分点,
                # 查不了就不许训 —— 旧代码 warning 后继续,等于门可以被
                # "H5 读不出来"这种最常见的故障静默绕过。
                raise RuntimeError(
                    f"[pose_mode] 无法校验 H5 位姿口径({h5}): {e} —— "
                    f"pose_mode 是三臂唯一区分点,校验不了不许开训。") from e
        if _is_main_process():
            logger.info(f"[pose_mode] {pose_mode}(H5 pose_injection={inj}, "
                        f"首样本位姿 identity={sample_identity})")
        if inj is None and sample_identity is None:
            return
        if inj is None:
            is_identity_h5 = sample_identity
            inj = f"(无 attr,按首样本位姿判定 identity={sample_identity})"
        else:
            is_identity_h5 = "identity" in inj
            if sample_identity is not None and sample_identity != is_identity_h5:
                raise RuntimeError(
                    f"[pose_mode] H5 自相矛盾:metadata 说 pose_injection={inj},"
                    f"但首样本位姿 identity={sample_identity} —— 文件被改名/篡改,"
                    f"commanded 语义不可信。")
        if pose_mode == "native" and is_identity_h5:
            raise ValueError(
                f"pose_mode=native 但数据是位姿中和的 H5(pose_injection={inj})"
                f" —— 几何臂拿不到真实轨迹。用 train_geo.h5,或改 --pose_mode。")
        if pose_mode == "identity" and not is_identity_h5:
            raise ValueError(
                f"pose_mode=identity 但数据不是中和 H5(pose_injection={inj})"
                f" —— 静止相机诊断臂必须配 neutralize 过的 H5。")

    def _assert_prompt_schema(self) -> None:
        """训练声明必须与 H5 实际 prompt schema 一致，official 缺字段即拒。

        prompt 是 drop/text-only 臂唯一的轨迹命令来源。若运行环境写
        ``net_continuous``、H5 却仍是旧四方向 ``net``，训练本身不会崩，
        manifest 还会把旧数据错误认证为连续轨迹实验，因此必须在加载模型前校验。
        """
        import h5py

        h5 = Path(self.args.data_path)
        try:
            with h5py.File(h5, "r") as f:
                raw = (f["metadata"].attrs.get("prompt_schema")
                       if "metadata" in f else None)
                if raw is not None:
                    self._data_prompt_schema = (
                        raw.decode() if isinstance(raw, bytes) else str(raw))
        except Exception as e:
            raise RuntimeError(
                f"[prompt_schema] 无法读取 {h5} 的数据口径: {e}") from e

        declared = os.environ.get("RSWM_PROMPT_SCHEMA")
        profile = os.environ.get("RSWM_PROFILE", "unset")
        if self._data_prompt_schema is None:
            if profile == "official":
                raise RuntimeError(
                    f"[prompt_schema] official 训练要求 H5 metadata.prompt_schema，"
                    f"但 {h5} 未记录；无法证明数据是连续 prompt 口径。请重建数据。")
            logger.warning("[prompt_schema] H5 未记录 schema（legacy 兼容，仅诊断可用）")
            return
        if declared and declared != self._data_prompt_schema:
            raise ValueError(
                f"[prompt_schema] 环境声明={declared}，H5 实际="
                f"{self._data_prompt_schema} —— 拒绝把旧 prompt 数据标成新口径。")
        logger.info("[prompt_schema] H5=%s，声明=%s",
                    self._data_prompt_schema, declared or "<未设>")

    def _prepare_plucker_embeddings(self, batch: Dict) -> Optional[Dict]:
        video = batch["video_frames"]
        B, F_total, C, H, W = video.shape

        vae_stride = self.config.vae_stride
        lat_h = H // vae_stride[1]
        lat_w = W // vae_stride[2]
        lat_f = (F_total - 1) // vae_stride[0] + 1

        Ks_raw = batch["intrinsics"]
        c2ws_raw = batch["poses"]

        if Ks_raw.dim() == 2:
            Ks = Ks_raw.unsqueeze(0).expand(B, -1, -1)
        elif Ks_raw.dim() == 3:
            Ks = Ks_raw
        else:
            raise ValueError(f"Unexpected intrinsics shape: {Ks_raw.shape}")

        if c2ws_raw.dim() == 3:
            c2ws = c2ws_raw.unsqueeze(0).expand(B, -1, -1, -1)
        elif c2ws_raw.dim() == 4:
            c2ws = c2ws_raw
        else:
            raise ValueError(f"Unexpected poses shape: {c2ws_raw.shape}")

        actual_T = batch.get("num_frames", F_total)
        Ks = Ks[:, :actual_T]
        c2ws = c2ws[:, :actual_T]

        dit_cond_list = []
        for i in range(B):
            Ks_i = Ks[i].float()
            c2ws_i = c2ws[i].float()
            # =================================================================
            # [v2.8 P0 时间轴修复] 旧行为 c2ws_i[:lat_f] 是【截断】不是重采样:
            # lat_f = (T-1)//4+1,对 17/25/33 帧分别只取前 5/7/9 个 pose ——
            # 实测三种配置下 pose 条件全部落在第一段(0..8)之内,**一个转弯点
            # 都不含**,只覆盖轨迹前 27-29%;而 inference.py:259-263 用
            # interpolate_camera_poses 把【完整】轨迹均匀重采样到 lat_f。
            # 训练/推理条件分布因此根本不一致 —— "几何通道零增益且损保真"、
            # "模型忽略转弯" 都可由此平凡解释。现两侧共用同一重采样口径。
            # =================================================================
            # [v2.8.2 N1] 内参与 c2ws 必须同口径。旧写法对 Ks 取【前 lat_f 帧】
            # 而 c2ws 是【全轨迹重采样】—— 当前内参是逐帧常数 dummy 所以数值无害,
            # 但这正是 v2.8 刚修掉的"截断 vs 重采样"错误在内参上的残留:
            # 一旦内参改成逐帧变化就是静默错。
            # [v2.9.1] 训练侧按 linspace 重采样,而 inference.py:263 是
            # `Ks = Ks[0]` + repeat(lat_f)(复制首帧)。内参目前逐帧恒定,
            # 两者逐位相同;一旦内参变成逐帧变化,两侧就会静默分叉。
            # 与其猜哪边对,不如让"前提失效"这件事本身可见。
            if Ks_i.shape[0] > 1 and not torch.allclose(
                    Ks_i, Ks_i[:1].expand_as(Ks_i), atol=1e-6):
                raise RuntimeError(
                    "内参不再逐帧恒定 —— 训练侧按 linspace 重采样,推理侧"
                    "(inference.py:263)复制首帧,两者会静默分叉。"
                    "改内参前必须先把两侧口径统一。")
            if Ks_i.shape[0] >= lat_f:
                _ki = np.round(np.linspace(0, Ks_i.shape[0] - 1, lat_f)).astype(int)
                Ks_i = Ks_i[_ki]
            # [v2.8.2] 这里【不】先重采样。原有的那一次 interpolate_camera_poses
            # (下方 c2ws_infer)直接吃完整轨迹即可 —— 与 inference.py:266-272
            # 逐字同式。v2.8 的第一版修复在这里额外插了一次重采样,结果训练侧
            # 变成"两次插值"而推理侧只有一次:多出的那趟 Slerp 走了四元数往返,
            # 不是逐位无损,于是刚修好的训推一致性又被自己引入了新的不对称。
            if c2ws_i.shape[0] < 2:
                raise ValueError(
                    f"pose 序列长度 {c2ws_i.shape[0]} < 2,无法重采样到 lat_f={lat_f}")

            Ks_trans = get_Ks_transformed(
                Ks_i,
                height_org=H,
                width_org=W,
                height_resize=H,
                width_resize=W,
                height_final=H,
                width_final=W,
            )

            if Ks_trans.shape[0] < lat_f:
                pad_n = lat_f - Ks_trans.shape[0]
                Ks_trans = torch.cat(
                    [Ks_trans, Ks_trans[-1:].expand(pad_n, -1)], dim=0
                )
            Ks_expand = Ks_trans[:lat_f].to(self.device)

            len_c2ws = c2ws_i.shape[0]
            if len_c2ws < lat_f:
                c2ws_i = torch.cat(
                    [c2ws_i, c2ws_i[-1:].expand(lat_f - len_c2ws, -1, -1)], dim=0
                )
                len_c2ws = lat_f

            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws_i[:, :3, :3].cpu().numpy(),
                src_trans_vec=c2ws_i[:, :3, 3].cpu().numpy(),
                tgt_indices=np.linspace(0, len_c2ws - 1, lat_f),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True).to(
                self.device
            )

            plucker = get_plucker_embeddings(c2ws_infer, Ks_expand, H, W)
            plucker = rearrange(
                plucker,
                "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
                c1=int(H // lat_h),
                c2=int(W // lat_w),
            )
            plucker = plucker[None, ...]
            plucker = rearrange(
                plucker, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w
            )
            dit_cond_list.append(plucker)

        c2ws_plucker_emb = torch.cat(dit_cond_list, dim=0).to(dtype=torch.bfloat16)
        return {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(B, dim=0)}

    # ----------------------------------------------------------- y condition
    def _build_y_conditioning(
        self, image: torch.Tensor, T: int, H: int, W: int, B: int
    ):
        vae_stride = self.config.vae_stride
        lat_h, lat_w = H // vae_stride[1], W // vae_stride[2]
        lat_f = (T - 1) // vae_stride[0] + 1

        assert (T - 1) % vae_stride[0] == 0, (
            f"T={T} 不满足 4n+1 规则 (vae_stride[0]={vae_stride[0]})。"
            f"请检查 _collate_fn 是否正确对齐帧数。"
        )

        y_list = []
        msk = torch.ones(1, T, lat_h, lat_w, device=self.device, dtype=torch.float32)
        msk[:, 1:] = 0
        msk = torch.cat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

        for i in range(B):
            img = image[i : i + 1]
            img_resized = torch.nn.functional.interpolate(
                img, size=(H, W), mode="bicubic"
            )
            img_first = img_resized.squeeze(0).unsqueeze(1)
            img_frames = torch.cat(
                [
                    img_first,
                    torch.zeros(3, T - 1, H, W, device=self.device, dtype=img.dtype),
                ],
                dim=1,
            )
            y_vae = self._vae_encode([img_frames.to(dtype=torch.bfloat16)])[0]
            y_i = torch.cat([msk.unsqueeze(0).to(y_vae.dtype), y_vae.unsqueeze(0)], dim=1)[0]
            y_list.append(y_i)
            del img_frames, y_vae
        del msk
        return y_list

    # ----------------------------------------------------------- compute loss
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        video = batch["video_frames"]
        image = batch["image"]
        B, T, C, H, W = video.shape

        # A800 80GB:先释放上一批仍驻 GPU 的 DiT，再换入 VAE/T5。上一批 backward
        # 已在调用 compute_loss 前完成；若处于梯度累积，Parameter.grad 与 AdamW
        # state 会由 _offload_to_cpu 一并迁移，因此不会丢梯度。
        if self.args.inactive_model_cpu:
            self._offload_all_experts()

        if (T - 1) % self.config.vae_stride[0] != 0:
            T_aligned = ((T - 1) // self.config.vae_stride[0]) * self.config.vae_stride[0] + 1
            video = video[:, :T_aligned]
            T = T_aligned

        vae_stride = self.config.vae_stride
        patch_size = self.config.patch_size
        lat_h = H // vae_stride[1]
        lat_w = W // vae_stride[2]
        lat_f = (T - 1) // vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])

        with torch.no_grad():
            video_list = [
                video[i].permute(1, 0, 2, 3).to(dtype=torch.bfloat16, device=self.device)
                for i in range(B)
            ]
            latent_list = self._vae_encode(video_list)
            latents = torch.stack(latent_list, dim=0)
            del video_list, latent_list

        with torch.no_grad():
            y_list = self._build_y_conditioning(image, T, H, W, B)

        timesteps = torch.randint(
            0, self.config.num_train_timesteps, (B,), device=self.device
        ).long()

        noise = torch.randn_like(latents, dtype=torch.bfloat16)
        t_frac = (
            timesteps.view(B, 1, 1, 1, 1).float() / self.config.num_train_timesteps
        )
        noisy_latents = (1 - t_frac.to(latents)) * latents + t_frac.to(latents) * noise
        noisy_latents = noisy_latents.to(torch.bfloat16)

        with torch.no_grad():
            if self.args.t5_cpu:
                self.text_encoder.model.to(self.device)
                torch.cuda.synchronize()
            context = self.text_encoder(batch["prompt"], self.device)
            if self.args.t5_cpu:
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # [v2.8 ④ pose_mode 三态] 语义固定,冲突即报错:
        #   native   : 真实轨迹 Plücker(几何臂)
        #   identity : 恒等位姿 = 【静止相机诊断臂】—— 注意它不是 "text-only",
        #              而是对相机条件预训练的骨干【显式命令"相机不动"】,与
        #              prompt 的方向词矛盾;旧稿把它当 text-only 是主对比的
        #              污染源(2026-07-29 审查确认)。
        #   drop     : 位姿通路整体不传(dit_cond_dict=None)= 真正 no-pose /
        #              text-only 臂。
        _pm = getattr(self.args, "pose_mode", "native")
        if not getattr(self, "_pose_mode_checked", False):
            self._assert_pose_mode(_pm)
            self._pose_mode_checked = True
        if _pm == "drop":
            dit_cond_dict = None
        else:
            dit_cond_dict = self._prepare_plucker_embeddings(batch)

        x_list = [noisy_latents[i] for i in range(B)]
        y_list_final = y_list

        amp_dtype = None
        if self.args.mixed_precision == "fp16":
            amp_dtype = torch.float16
        elif self.args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16

        low_mask = timesteps < self.boundary_step
        high_mask = ~low_mask

        if self.args.inactive_model_cpu and B != 1:
            raise RuntimeError(
                "--inactive_model_cpu 要求 batch_size=1；同一 batch 同时命中 low/high "
                "专家时，前向图不能在 backward 前安全换出。A800 80GB 请保持 "
                "--batch_size 1，并用 --gradient_accumulation_steps 保持有效 batch。"
            )

        pred_list = [None] * B

        sub_seq = max_seq_len
        for tag, mask, model in [
            ("low", low_mask, self.low_noise_model),
            ("high", high_mask, self.high_noise_model),
        ]:
            indices = mask.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue

            if self.args.inactive_model_cpu:
                self._ensure_on_gpu(tag)

            sub_x = [x_list[i] for i in indices]
            sub_t = timesteps[indices]
            sub_ctx = [context[i] for i in indices]
            sub_y = [y_list_final[i] for i in indices]
            # [v2.8.2 P0] pose_mode=drop 时 dit_cond_dict 就是 None(真·无位姿),
            # 这里必须跟着传 None。旧写法无条件下标 → 第一个 batch 就
            # TypeError: 'NoneType' object is not subscriptable,34/35/36 三个
            # 作业活不过一分钟,而 20-33 全部 afterok:34 → 整条链卡死、产出为零。
            # (v2.8 之前 _prepare_plucker_embeddings 从不返回 None,这条路径不可达
            #  —— 是 drop 分支自己引入的。)DiT 侧本就有守卫:
            #  wan/modules/model.py:257 与 :509 都是 `dit_cond_dict is not None`,
            #  wan/image2video.py:328 也用 None 表示"不注入位姿"。
            sub_plucker = None
            if dit_cond_dict is not None:
                sub_plucker = {
                    "c2ws_plucker_emb": [dit_cond_dict["c2ws_plucker_emb"][i]
                                         for i in indices]
                }

            fwd_dtype = amp_dtype if amp_dtype is not None else torch.bfloat16
            with torch.amp.autocast("cuda", dtype=fwd_dtype):
                sub_pred = model(
                    x=sub_x,
                    t=sub_t,
                    context=sub_ctx,
                    seq_len=sub_seq,
                    y=sub_y,
                    dit_cond_dict=sub_plucker,
                )

            for j, idx in enumerate(indices):
                pred_list[idx] = sub_pred[j]

        pred = torch.stack(pred_list, dim=0)
        target = (noise - latents).float()
        loss = F.mse_loss(pred.float(), target, reduction="mean")

        del y_list, y_list_final, x_list, noisy_latents, dit_cond_dict, context
        return loss

    # ----------------------------------------------------------- optimizer
    def _init_optimizer(self):
        params = list(self.low_noise_model.parameters()) + list(
            self.high_noise_model.parameters()
        )
        params = [p for p in params if p.requires_grad]
        self.optimizer = _build_optimizer(
            params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            use_8bit=self.args.use_8bit_adam,
        )
        steps_per_epoch = max(
            1,
            (len(self.train_loader) + self.args.gradient_accumulation_steps - 1)
            // self.args.gradient_accumulation_steps,
        )
        total_steps = steps_per_epoch * self.args.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.min_lr,
        )

    # ----------------------------------------------------------- checkpoint
    def _load_resume_lora_weights(self, checkpoint_path: str):
        ckpt_dir = Path(checkpoint_path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
        logger.info(f"从 {checkpoint_path} 加载 LoRA 权重...")
        for subdir, model in [
            ("low_noise", self.low_noise_model),
            ("high_noise", self.high_noise_model),
        ]:
            path = ckpt_dir / subdir
            if not path.exists():
                continue
            adapter_file = path / "adapter_model.safetensors"
            if not adapter_file.exists():
                adapter_file = path / "adapter_model.bin"
            if adapter_file.exists():
                state = self._load_adapter_state(str(adapter_file))
                set_peft_model_state_dict(model, state)
                logger.info(f"已加载 LoRA: {subdir}")

    def _build_run_manifest(self) -> dict:
        """[v2.9.2] 本次训练的完整身份。续训、评测复用、汇总三处共用同一份字段。"""
        from run_manifest import build_manifest, sha256_file
        a = self.args
        # manifest 绑定训练器最终实际采用的两路视图（含 cap 后索引）。
        # 原始 official 三路文件另以 source_split_file/source_split_sha256 留在
        # extra，既能证明 test 未进训练，也避免训练 checkpoint 与 eval 的
        # val-only split 因文件格式不同而无法比较。
        _split_json = (Path(getattr(a, "output_dir", ".")) /
                       "train_val_split.json")
        _source_split = (Path(a.split_file) if getattr(a, "split_file", None)
                         else None)
        return build_manifest(
            role=os.environ.get("RSWM_RUN_ROLE", "train"),
            pose_mode=getattr(a, "pose_mode", "native"),
            data_h5=getattr(a, "data_path", None),
            split_file=_split_json if _split_json.is_file() else None,
            checkpoint_id=str(getattr(a, "checkpoint_dir", "")),
            prompt_schema=self._data_prompt_schema,
            frame_num=getattr(a, "max_frames", None),
            train_seed=getattr(a, "seed", None),
            split_seed=(getattr(a, "split_seed", None)
                        if getattr(a, "split_seed", None) is not None
                        else getattr(a, "seed", None)),
            dataset_tier=os.environ.get("DATASET_TIER"),
            lora_rank=getattr(a, "lora_rank", None),
            lora_alpha=getattr(a, "lora_alpha", None),
            lora_targets=getattr(a, "lora_targets", None),
            max_train_samples=getattr(a, "max_train_samples", None),
            num_epochs=getattr(a, "num_epochs", None),
            batch_size=getattr(a, "batch_size", None),
            gradient_accumulation_steps=getattr(
                a, "gradient_accumulation_steps", None),
            learning_rate=getattr(a, "learning_rate", None),
            extra={
                "source_split_file": (str(_source_split)
                                      if _source_split is not None else None),
                "source_split_sha256": (
                    sha256_file(_source_split)
                    if _source_split is not None else None),
                # 硬件/优化器不参与推理侧 manifest 配对，但必须随训练产物留痕，
                # 便于复现实验并区分 A800 offload 与 H200 常驻路径。
                "hardware_profile": getattr(a, "hardware_profile", "auto"),
                "optimizer_impl": (
                    "bitsandbytes.AdamW8bit" if getattr(a, "use_8bit_adam", False)
                    else "torch.optim.AdamW"),
                "inactive_model_cpu": bool(getattr(a, "inactive_model_cpu", False)),
                "offload_requested": getattr(a, "offload_requested", "auto"),
                "offload_resolved": getattr(a, "offload_resolved", None),
                "offload_resolution_reason": getattr(
                    a, "offload_resolution_reason", None),
                "mixed_precision": getattr(a, "mixed_precision", None),
                "gradient_checkpointing": bool(
                    getattr(a, "gradient_checkpointing", False)),
                "attention_backend": getattr(a, "attention_backend", "auto"),
                "num_workers": getattr(a, "num_workers", None),
                "prefetch_factor": getattr(a, "prefetch_factor", None),
                "persistent_workers": bool(
                    getattr(a, "persistent_workers", False)),
                "pin_memory": bool(getattr(a, "pin_memory", False)),
                "torch_compile": bool(getattr(a, "torch_compile", False)),
                "hardware_profile_config": getattr(
                    a, "hardware_profile_config", None),
                "max_val_samples": getattr(a, "max_val_samples", None),
                "runtime_gpu": getattr(a, "runtime_gpu_info", None),
            },
        )

    def _assert_resume_pose_mode(self, checkpoint_path: str) -> None:
        """[v2.8.2 P0/C4] 续训必须同 pose 口径,否则静默跨 regime 训练。

        train_lora 会自动扫 `$out/checkpoint-epoch-*` 续训。只要 $out 里留着
        pre-v2.8 的旧断点(20-33 那批"全部作废"的),或把 SEM_ARM_MODE 从
        identity 翻成 drop 后重投同名作业,就会用【新口径】接着训【旧口径】的
        权重 —— epoch 1..k 与 k+1..N 分属两个 pose regime。
        更隐蔽的是 save_checkpoint 会用【新的】pose_mode 覆写 training_config.json,
        于是 eval_lora 的交叉校验照样 PASS。v2.8 要根治的"训练/推理口径不一致"
        以更难查的形式复活。
        """
        import json as _json
        cfg = Path(checkpoint_path) / "training_config.json"
        if not cfg.is_file():
            logger.warning(f"[resume] {cfg} 不存在,无法校验 pose_mode —— "
                           f"若这是 v2.8 之前的断点,请勿续训。")
            return
        try:
            _d = _json.loads(cfg.read_text())
        except Exception as e:
            raise RuntimeError(f"[resume] 无法解析 {cfg}: {e}") from e
        old = (_d.get("data_split", {}) or {}).get("pose_mode") or _d.get("pose_mode")
        new = getattr(self.args, "pose_mode", "native")
        if old is None:
            raise RuntimeError(
                f"[resume] 断点 {checkpoint_path} 没有记录 pose_mode(v2.8 之前的"
                f"产物)。它的 pose 时间轴是被截断的旧口径,与当前不可混训 —— "
                f"删除该目录重训,或显式确认后手工补写 training_config.json。")
        if old != new:
            raise RuntimeError(
                f"[resume] pose_mode 冲突:断点={old} 当前={new} —— 续训会让"
                f"前后 epoch 分属两个 pose regime,而 save_checkpoint 会用新值"
                f"覆写配置,下游任何一层都查不出来。换输出目录或删除旧断点。")
        # [v2.9] profile 漂移同样是静默污染:同一个 $out 下先按 legacy(identity)
        # 训了几个 epoch、再按 official(drop)续训,权重前后分属两个 regime,
        # 而 training_config.json 会被新值覆写 → 下游查不出来。
        # [v2.9.2 P0-1] 读【正确层级】。保存时 rswm_profile 落在
        # config_dict["data_split"] 里(与 pose_mode 同处),而旧代码读顶层 →
        # 恒为 None → 整条 profile 校验形同虚设。native 臂在 official/legacy
        # 下 pose_mode 都是 native,于是 legacy 断点能直接跨 profile 续训,
        # 随后被 save_checkpoint 重新标成 official。
        _ds = _d.get("data_split") or {}
        _op = _ds.get("rswm_profile") or _d.get("rswm_profile")
        _np_ = os.environ.get("RSWM_PROFILE", "unset")
        if _op is None or _op == "unset":
            raise RuntimeError(
                f"[resume] 断点 {checkpoint_path} 没有记录 rswm_profile —— "
                f"v2.9.2 之前的产物,无法证明它与本次 profile 同源。"
                f"缺字段一律拒绝(fail-closed):删除该目录重训。")
        if _np_ != "unset" and _op != _np_:
            raise RuntimeError(
                f"[resume] RSWM_PROFILE 冲突:断点={_op} 当前={_np_} —— "
                f"official(drop 主线)与 legacy(identity)的臂语义不同,"
                f"不可混训。换输出目录或删除旧断点。")
        # 全字段 manifest 校验:pose_mode/profile 只是其中两项,data_h5、
        # split、代码指纹变了同样不许接着训。
        from run_manifest import (read_manifest, assert_compatible,
                                  summarize)
        _prev = read_manifest(checkpoint_path)
        assert_compatible(_prev, self._build_run_manifest(), ctx="续训")
        logger.info(f"[resume] 身份校验通过 — {summarize(_prev)}")

    def _load_resume_training_state(self, checkpoint_path: str):
        ckpt_dir = Path(checkpoint_path)
        state_file = ckpt_dir / "training_state.pt"
        if state_file.exists():
            # offload 模式下两个专家此刻都在 CPU。先把 state 加到 CPU，再由
            # _ensure_on_gpu 按专家搬运；直接 map 到 CUDA 会让 inactive 参数在
            # CPU、Adam state 在 GPU，下一次 optimizer.step device mismatch。
            resume_device = "cpu" if self.args.inactive_model_cpu else self.device
            state = torch.load(state_file, map_location=resume_device)
            self.start_epoch = state.get("epoch", 1) + 1
            if "optimizer" in state:
                try:
                    self.optimizer.load_state_dict(state["optimizer"])
                    if self.args.inactive_model_cpu:
                        self._move_optimizer_state(
                            self.low_noise_model, torch.device("cpu"))
                        self._move_optimizer_state(
                            self.high_noise_model, torch.device("cpu"))
                except Exception as e:
                    logger.warning(
                        f"optimizer state 加载失败（可能因为 AdamW8bit/AdamW 切换），跳过：{e}"
                    )
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            if "best_val_loss" in state:
                self.best_val_loss = state["best_val_loss"]
            logger.info(f"恢复至 epoch {self.start_epoch}")
        log_file = Path(self.args.output_dir) / "training_log.json"
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                self.training_log = json.load(f)

    def _load_adapter_state(self, path: str) -> dict:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        state = torch.load(path, map_location="cpu")
        return state.get("state_dict", state) if isinstance(state, dict) else state

    def _wrap_ddp(self):

        # [v2.7.1 F6 守卫] --inactive_model_cpu(默认开)把模型放 CPU,而

        # DDP(device_ids=[gpu]) 对 CPU 参数直接 ValueError —— torchrun 默认

        # 参数必崩。现役单卡 PBS 路径无 RANK 不走这里;真要多卡,须用

        # --no_inactive_model_cpu(store_true 型,不能接数值)关闭 offload。

        if getattr(self.args, "inactive_model_cpu", True):

            raise RuntimeError(

                "DDP 与 --inactive_model_cpu(默认开)不兼容:模型在 CPU 上时 "

                "DDP(device_ids=...) 会 ValueError。多卡请用 --no_inactive_model_cpu "

                "显式关闭 CPU offload(H200 141G 装得下两个 14B 专家)。")
        if _is_main_process():
            logger.info("使用 DDP 包装模型...")
        self.low_noise_model = DDP(
            self.low_noise_model, device_ids=[self.local_rank]
        )
        self.high_noise_model = DDP(
            self.high_noise_model, device_ids=[self.local_rank]
        )

    def _get_raw_model(self, model):
        return model.module if self.use_ddp else model

    # ------------------------------------------------------------- training
    def _optimizer_step(self):
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.low_noise_model.parameters() if p.requires_grad]
                + [p for p in self.high_noise_model.parameters() if p.requires_grad],
                self.args.max_grad_norm,
            )
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        # optimizer.step 后没有待反传的图，可以把当前专家也换回 CPU；这样验证、
        # 保存 checkpoint 或下一批的 VAE/T5 都不会与 14B DiT 争抢 A800 显存。
        if self.args.inactive_model_cpu:
            self._offload_all_experts()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.low_noise_model.train()
        self.high_noise_model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = self.args.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.args.num_epochs}",
            disable=not _is_main_process(),
        )

        step = -1
        for step, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)

            loss = self.compute_loss(batch) / accum_steps

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum_steps == 0:
                self._optimizer_step()

            total_loss += loss.item() * accum_steps
            n_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item() * accum_steps:.4f}",
                avg=f"{total_loss / n_batches:.4f}",
            )
            del batch, loss

        if n_batches > 0 and (step + 1) % accum_steps != 0:
            self._optimizer_step()

        return {"loss": total_loss / n_batches if n_batches else 0.0}

    def validate(self, epoch: int) -> Dict[str, float]:
        if len(self.val_dataset) == 0:
            return {"loss": float("inf")}
        self.low_noise_model.eval()
        self.high_noise_model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, desc="Validation", disable=not _is_main_process()
            ):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                n_batches += 1
                del batch, loss

        if self.args.inactive_model_cpu:
            self._offload_all_experts()

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_loss, float(n_batches)], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            n_batches = int(loss_tensor[1].item())

        return {"loss": total_loss / n_batches if n_batches else 0.0}

    def save_checkpoint(
        self, epoch: int, train_metrics: Dict, val_metrics: Dict, append_log: bool = True
    ):
        if self.use_ddp:
            dist.barrier()
        if not _is_main_process():
            return None

        save_dir = Path(self.args.output_dir) / (
            f"checkpoint-epoch-{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        # 先写本次实际 train/val 索引，再建 manifest；否则首个 checkpoint
        # 生成时 output_dir/train_val_split.json 尚不存在，split_sha256 会是 None，
        # 后续 epoch/续训看到文件存在又变成真 hash，身份会自相矛盾。
        split_path = Path(self.args.output_dir) / "train_val_split.json"
        if not split_path.is_file():
            raise RuntimeError(
                f"保存 checkpoint 前缺 {split_path}；无法把实际划分绑定进 manifest")
        # [v2.9.2] 每个 checkpoint 自带完整身份。续训与评测都据此校验,
        # 不再靠"training_config.json 里某个字段"这种单点比对。
        from run_manifest import write_manifest
        write_manifest(save_dir, self._build_run_manifest())

        low_model = self._get_raw_model(self.low_noise_model)
        high_model = self._get_raw_model(self.high_noise_model)

        (save_dir / "low_noise").mkdir(exist_ok=True)
        (save_dir / "high_noise").mkdir(exist_ok=True)
        low_model.save_pretrained(save_dir / "low_noise")
        high_model.save_pretrained(save_dir / "high_noise")

        state = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": "i2v-A14B",
            "lora_rank": self.args.lora_rank,
            "lora_alpha": self.args.lora_alpha,
            "use_8bit_adam": self.args.use_8bit_adam,
            "hardware_profile": self.args.hardware_profile,
            "inactive_model_cpu": self.args.inactive_model_cpu,
            "offload_requested": self.args.offload_requested,
            "offload_resolved": self.args.offload_resolved,
        }
        torch.save(state, save_dir / "training_state.pt")

        config_dict = {
            "epoch": epoch,
            "checkpoint_dir": self.args.checkpoint_dir,
            "data_path": self.args.data_path,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lora_rank": self.args.lora_rank,
            "lora_alpha": self.args.lora_alpha,
            "lora_targets": self.args.lora_targets,
            "hardware_profile": self.args.hardware_profile,
            "runtime_gpu": getattr(self.args, "runtime_gpu_info", None),
            "optimizer_impl": (
                "bitsandbytes.AdamW8bit" if self.args.use_8bit_adam
                else "torch.optim.AdamW"),
            "inactive_model_cpu": self.args.inactive_model_cpu,
            "offload_requested": self.args.offload_requested,
            "offload_resolved": self.args.offload_resolved,
            "offload_resolution_reason": self.args.offload_resolution_reason,
            "mixed_precision": self.args.mixed_precision,
            "gradient_checkpointing": self.args.gradient_checkpointing,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "attention_backend": self.args.attention_backend,
            "num_workers": self.args.num_workers,
            "prefetch_factor": self.args.prefetch_factor,
            "persistent_workers": self.args.persistent_workers,
            "pin_memory": self.args.pin_memory,
            "torch_compile": self.args.torch_compile,
            "hardware_profile_config": self.args.hardware_profile_config,
            "max_train_samples": self.args.max_train_samples,
            "max_val_samples": self.args.max_val_samples,
            "prompt_schema": self._data_prompt_schema,
            # 参数效率审计 (advisor point 8): 同样折进 checkpoint config,
            # 让 eval/分析端不依赖 output_dir/param_audit.json 也能还原。
            "param_audit": getattr(self, "param_audit", None),
            "data_split": {
                "train_ratio": self.args.train_ratio,
                "val_ratio": self.args.val_ratio,
                "train_samples": len(self.train_dataset),
                "val_samples": len(self.val_dataset),
                "train_indices": self.train_indices,
                "val_indices": self.val_indices,
                # [v2.7.1 F8/P2] 划分实际用 split_seed(多种子实验共享 42),
                # 旧字段记训练 seed → JSON 显示 43/44 误导复现审计
                "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
                "headline_eligible": os.environ.get("RSWM_HEADLINE_ELIGIBLE", "0") == "1",
                "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
                "pose_mode": getattr(self.args, "pose_mode", "native"),  # [v2.8 ④]
                "train_seed": self.args.seed,
                "split_seed": (self.args.split_seed
                               if getattr(self.args, "split_seed", None) is not None
                                else self.args.seed),
                "max_train_samples": self.args.max_train_samples,
                "max_val_samples": self.args.max_val_samples,
                "prompt_schema": self._data_prompt_schema,
                "split_file": self.args.split_file,
                "source_split_sha256": (
                    hashlib.sha256(Path(self.args.split_file).read_bytes()).hexdigest()
                    if self.args.split_file else None),
            },
        }
        with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        if append_log:
            self._append_training_log(epoch, train_metrics, val_metrics)
        logger.info(f"Checkpoint 已保存: {save_dir}")
        return save_dir

    def _append_training_log(self, epoch, train_metrics, val_metrics):
        record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        self.training_log.append(record)
        log_path = Path(self.args.output_dir) / "training_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------------- main
    def train(self):
        if _is_main_process():
            logger.info("开始训练...")
        start_time = time.time()
        # 重置显存峰值计数器,使 training_summary.json 里的 peak_gpu_mem_* 只统计
        # 【训练本身】,不含模型加载/VAE-T5 swap 的瞬时峰值(参数效率表要的是训练开销)。
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass  # 统计是尽力而为,绝不因此中断训练
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if _is_main_process():
            split_info = {
                "train_ratio": self.args.train_ratio,
                "val_ratio": self.args.val_ratio,
                "train_samples": len(self.train_dataset),
                "val_samples": len(self.val_dataset),
                "train_indices": self.train_indices,
                "val_indices": self.val_indices,
                # [v2.7.1 F8/P2] 同上:split 溯源必须记真实划分种子
                "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
                "headline_eligible": os.environ.get("RSWM_HEADLINE_ELIGIBLE", "0") == "1",
                "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
                "pose_mode": getattr(self.args, "pose_mode", "native"),  # [v2.8 ④]
                "train_seed": self.args.seed,
                "split_seed": (self.args.split_seed
                               if getattr(self.args, "split_seed", None) is not None
                               else self.args.seed),
                # 训练集上限和 prompt 口径都是数据身份的一部分。旧版只写索引，
                # 事后无法判断这是全量训练还是 smoke/capped run，也无法证明
                # 训练时实际使用的是 net 还是 net_continuous prompt。
                "max_train_samples": (
                    int(self.args.max_train_samples)
                    if self.args.max_train_samples is not None else None),
                "max_val_samples": (
                    int(self.args.max_val_samples)
                    if self.args.max_val_samples is not None else None),
                "prompt_schema": self._data_prompt_schema,
                "split_file": self.args.split_file,
                "source_split_sha256": (
                    hashlib.sha256(Path(self.args.split_file).read_bytes()).hexdigest()
                    if self.args.split_file else None),
            }
            with open(out_dir / "train_val_split.json", "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2, ensure_ascii=False)
            logger.info(f"训练/验证划分已保存至 {out_dir / 'train_val_split.json'}")

        train_metrics: Dict = {"loss": 0.0}
        val_metrics: Dict = {"loss": float("inf")}
        last_epoch = self.start_epoch

        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            last_epoch = epoch
            if _interrupt_requested:
                logger.info("检测到中断请求，保存当前状态...")
                self.save_checkpoint(
                    epoch - 1 if epoch > 1 else 1,
                    train_metrics,
                    val_metrics,
                    append_log=False,
                )
                break

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            if _is_main_process():
                logger.info(
                    f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}"
                )

            # [v2.7.1 F4/P1 修复] 最佳判定必须在 save_checkpoint 【之前】:
            # 旧顺序先保存(training_state.pt 里写进【未更新】的 best_val_loss,
            # epoch1 时=inf)再更新最佳值 → 从该 checkpoint 恢复后,val=0.6 也会
            # 被判"优于 inf",rmtree 掉真最佳(0.5)的 best_checkpoint 并覆盖。
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            ckpt_dir = self.save_checkpoint(epoch, train_metrics, val_metrics)

            if is_best:
                if _is_main_process():
                    import shutil

                    best_dir = Path(self.args.output_dir) / "best_checkpoint"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                    # [v2.7.2 ③/P1 修复] 直接用 save_checkpoint 返回的目录。
                    # 旧写法 sorted(glob("checkpoint-epoch-*"))[-1] 是字符串排序:
                    # epoch-9 > epoch-10,第 10 轮最佳会复制第 9 轮进 best_checkpoint
                    # (默认 10 epochs 正好触发;服务器 5 epochs 侥幸躲过)。
                    shutil.copytree(ckpt_dir, best_dir)
                    logger.info(
                        f"新的最佳模型 (val_loss={self.best_val_loss:.4f})"
                    )

        if _is_main_process():
            summary = {
                "total_epochs": last_epoch,
                "best_val_loss": self.best_val_loss,
                "total_time_seconds": time.time() - start_time,
                "hardware_profile": self.args.hardware_profile,
                "runtime_gpu": getattr(self.args, "runtime_gpu_info", None),
                "optimizer_impl": (
                    "bitsandbytes.AdamW8bit" if self.args.use_8bit_adam
                    else "torch.optim.AdamW"),
                "inactive_model_cpu": self.args.inactive_model_cpu,
                "offload_requested": self.args.offload_requested,
                "offload_resolved": self.args.offload_resolved,
                "offload_resolution_reason": self.args.offload_resolution_reason,
                "attention_backend": self.args.attention_backend,
                "num_workers": self.args.num_workers,
                "prefetch_factor": self.args.prefetch_factor,
                "persistent_workers": self.args.persistent_workers,
                "torch_compile": self.args.torch_compile,
                "max_train_samples": self.args.max_train_samples,
                "max_val_samples": self.args.max_val_samples,
            }
            # 峰值显存 —— 参数效率表要的实测字段(advisor point 8)。
            # torch 自带累计计数器,取值零开销;拿不到就写 null,绝不臆造。
            # peak_gpu_mem_gb 是本进程自训练开始以来的最大值(reset 见 _init_models 之前)。
            if torch.cuda.is_available():
                try:
                    dev = self.device
                    summary["peak_gpu_mem_allocated_gb"] = round(
                        torch.cuda.max_memory_allocated(dev) / 1024 ** 3, 3
                    )
                    summary["peak_gpu_mem_reserved_gb"] = round(
                        torch.cuda.max_memory_reserved(dev) / 1024 ** 3, 3
                    )
                except Exception as e:  # 统计失败不该拖垮训练收尾
                    logger.warning(f"峰值显存统计失败,记 null: {type(e).__name__}: {e}")
                    summary["peak_gpu_mem_allocated_gb"] = None
                    summary["peak_gpu_mem_reserved_gb"] = None
            else:
                summary["peak_gpu_mem_allocated_gb"] = None
                summary["peak_gpu_mem_reserved_gb"] = None
            with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(
                f"训练完成，总用时 {time.time() - start_time:.1f}s; "
                f"峰值显存 allocated={summary['peak_gpu_mem_allocated_gb']} GB "
                f"reserved={summary['peak_gpu_mem_reserved_gb']} GB"
            )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def _load_hardware_profile_spec(name: str) -> dict:
    if name == "auto":
        return {}
    path = HERE / "configs" / "hardware" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"hardware profile 不存在: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("profile") != name:
        raise ValueError(f"hardware profile 文件身份不一致: {path}")
    return spec


def _resolve_offload_mode(args, spec: dict):
    """Resolve OFFLOAD=auto/0/1 before loading either 14B expert.

    ``auto`` is deliberately conservative: resident mode is selected only
    when the *visible logical GPU* has both >=97,000 MiB total and >=90,000
    MiB currently free.  This keeps A800-80GB on the safe swap path while
    allowing an empty H200-class GPU to use the resident path.  Explicit 0/1
    remains available for controlled throughput/offload comparisons.
    """
    cli_mode = getattr(args, "offload", None)
    raw_mode = (cli_mode if cli_mode is not None else os.environ.get(
        "OFFLOAD", spec.get("offload", "auto")))
    raw_mode = str(raw_mode).strip().lower()

    # Backward-compatible aliases.  argparse uses None as the default so an
    # actually supplied legacy flag can override auto without ambiguity.
    legacy_flag = getattr(args, "inactive_model_cpu", None)
    if legacy_flag is not None:
        legacy_mode = "1" if bool(legacy_flag) else "0"
        if cli_mode is not None and raw_mode not in ("auto", legacy_mode):
            raise ValueError(
                f"--offload={raw_mode} conflicts with legacy offload flag "
                f"(resolved {legacy_mode})")
        raw_mode = legacy_mode

    if raw_mode not in ("auto", "0", "1"):
        raise ValueError("--offload/OFFLOAD must be one of auto, 0, 1")

    resolved = raw_mode
    reason = "explicit"
    total_mib = None
    free_mib = None
    if raw_mode == "auto":
        resolved = "1"
        reason = "no visible CUDA GPU; conservative pre-validation default"
        if torch.cuda.is_available():
            device_id = (int(os.environ["LOCAL_RANK"])
                         if "LOCAL_RANK" in os.environ
                         else (int(args.device_id)
                               if getattr(args, "device_id", None) is not None
                               else torch.cuda.current_device()))
            props = torch.cuda.get_device_properties(device_id)
            total_mib = int(props.total_memory / 1024 ** 2)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device_id)
                free_mib = int(free_bytes / 1024 ** 2)
            except (RuntimeError, TypeError):
                free_mib = None
            enough_total = total_mib >= 97000
            enough_free = free_mib is not None and free_mib >= 90000
            resolved = "0" if enough_total and enough_free else "1"
            reason = (
                f"visible logical GPU total={total_mib}MiB free="
                f"{free_mib if free_mib is not None else 'unknown'}MiB; "
                "resident thresholds total>=97000MiB and free>=90000MiB")

    args.offload_requested = raw_mode
    args.offload_resolved = resolved
    args.offload_resolution_reason = reason
    args.inactive_model_cpu = resolved == "1"
    return args


def _apply_hardware_profile(args):
    """把硬件 profile 解析成可审计的运行参数。"""
    spec = _load_hardware_profile_spec(args.hardware_profile)
    args.hardware_profile_config = spec
    args = _resolve_offload_mode(args, spec)
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = int(
            spec.get("gradient_accumulation_steps", 8))
    if args.num_workers is None:
        args.num_workers = int(spec.get("num_workers", 0))
    if args.prefetch_factor is None:
        args.prefetch_factor = spec.get("prefetch_factor")
    if args.prefetch_factor is None and args.num_workers > 0:
        args.prefetch_factor = 2
    if args.attention_backend is None:
        args.attention_backend = str(
            spec.get("attention_backend", os.environ.get(
                "RSWM_ATTENTION_BACKEND", "auto")))
    args.pin_memory = bool(spec.get(
        "pin_memory", not (args.low_mem or args.vae_cpu)))
    args.torch_compile = bool(spec.get("torch_compile", False))

    if args.hardware_profile == "a800_80g":
        # A800 是 Ampere(sm80)，原生支持 bf16。auto 在 80GB 上解析为
        # 单专家交换；显式 OFFLOAD=0 仅用于用户批准的吞吐/OOM 边界测试。
        # A800 固定使用 torch AdamW，避免把未锁定的 bnb 引入正式环境。
        args.low_mem = True
        args.t5_cpu = True
        args.vae_cpu = True
        args.gradient_checkpointing = True
        args.mixed_precision = "bf16"
        args.use_8bit_adam = False
        args.persistent_workers = False
        args.pin_memory = False
        if not args.inactive_model_cpu:
            logger.warning(
                "A800 profile 显式 OFFLOAD=0：允许作为吞吐/OOM 边界测试，"
                "但两份 14B 专家常驻 80GB 很可能 OOM；在真实 Level 3 + "
                "33 帧 smoke 通过前不得用于正式矩阵。")
    elif args.hardware_profile == "h200_141g":
        args.mixed_precision = str(spec.get("precision", "bf16"))
        args.t5_cpu = bool(spec.get("t5_cpu", True))
        args.vae_cpu = bool(spec.get("vae_cpu", True))
        args.gradient_checkpointing = bool(
            spec.get("gradient_checkpointing", True))
        args.use_8bit_adam = (
            not args.inactive_model_cpu
            and spec.get("optimizer") == "bitsandbytes.AdamW8bit")

    if args.low_mem:
        args.t5_cpu = True
        args.vae_cpu = True
        args.gradient_checkpointing = True
        if args.mixed_precision == "no":
            args.mixed_precision = "bf16"

    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps 必须 >= 1")
    if args.hardware_profile == "a800_80g" and args.batch_size != 1:
        raise ValueError(
            "a800_80g profile requires --batch_size 1; use gradient "
            "accumulation to increase effective batch size.")
    if args.inactive_model_cpu and args.batch_size != 1:
        raise ValueError(
            "CPU expert offload 只支持 --batch_size 1；A800 80GB 请用梯度累积"
            "提高有效 batch size。")
    if args.inactive_model_cpu and args.use_8bit_adam:
        raise ValueError(
            "--inactive_model_cpu 与 bitsandbytes AdamW8bit 不兼容。A800 80GB "
            "请使用 --hardware_profile a800_80g（会选择 torch.AdamW），或显式传 "
            "--no_8bit_adam。")
    if args.num_workers == 0:
        args.prefetch_factor = None
        args.persistent_workers = False
    if args.prefetch_factor is not None and args.prefetch_factor < 1:
        raise ValueError("--prefetch_factor 必须 >=1，或在 num_workers=0 时省略")
    os.environ["RSWM_ATTENTION_BACKEND"] = args.attention_backend
    return args


def _validate_runtime_gpu(args) -> dict:
    """启动 28B 双专家前先核实实际 GPU，避免在错误节点白白加载权重。"""
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 CUDA GPU，但 torch.cuda.is_available()=False")
    device_id = (int(os.environ["LOCAL_RANK"])
                 if "LOCAL_RANK" in os.environ
                 else (int(args.device_id) if args.device_id is not None
                       else torch.cuda.current_device()))
    torch.cuda.set_device(device_id)
    props = torch.cuda.get_device_properties(device_id)
    total_gib = props.total_memory / 1024 ** 3
    free_bytes, _ = torch.cuda.mem_get_info(device_id)
    free_gib = free_bytes / 1024 ** 3
    major, minor = torch.cuda.get_device_capability(device_id)
    info = {
        "name": props.name,
        "total_memory_gib": round(total_gib, 2),
        "free_memory_before_model_load_gib": round(free_gib, 2),
        "compute_capability": f"{major}.{minor}",
        "device_id": device_id,
        "offload_requested": getattr(args, "offload_requested", "auto"),
        "offload_resolved": getattr(args, "offload_resolved", "1"),
        "offload_resolution_reason": getattr(
            args, "offload_resolution_reason", None),
    }

    if args.hardware_profile == "a800_80g":
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "a800_80g 是共享服务器单卡安全 profile，要求当前进程只看见 1 张卡；"
                f"torch 当前看见 {torch.cuda.device_count()} 张。请先由管理员分配 GPU，"
                "再设置 CUDA_VISIBLE_DEVICES=<单个物理卡号>。不会自动选择或占用其余卡。")
        if ("A800" not in props.name or major != 8
                or not (75.0 <= total_gib < 90.0)):
            raise RuntimeError(
                "a800_80g profile 与实际 GPU 不符："
                f"name={props.name!r}, VRAM={total_gib:.1f}GiB, sm={major}{minor}。"
                "该 profile 仅用于 NVIDIA A800 80GB(sm80) 单卡。")
    elif args.hardware_profile == "h200_141g":
        if major < 9 or total_gib < 120.0:
            raise RuntimeError(
                "h200_141g profile 与实际 GPU 不符："
                f"name={props.name!r}, VRAM={total_gib:.1f}GiB, sm={major}{minor}。")

    logger.info(
        "GPU 运行核实: profile=%s name=%s VRAM=%.2fGiB free=%.2fGiB "
        "sm=%d%d offload=%s->%s",
        args.hardware_profile, props.name, total_gib, free_gib, major, minor,
        getattr(args, "offload_requested", "auto"),
        getattr(args, "offload_resolved", "1"))
    return info


def parse_args():
    p = argparse.ArgumentParser(description="LingBot 遥感 LoRA 微调 (A800/A100 80GB 版)")
    p.add_argument("--checkpoint_dir", required=True, help="预训练模型目录")
    p.add_argument("--data_path", required=True, help="训练数据 HDF5")
    p.add_argument("--output_dir", default="./rs_lora_output", help="输出目录")
    p.add_argument("--resume_from", default=None, help="从 checkpoint 恢复")

    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--max_frames", type=int, default=33,
                   help="正式 RSRoll/RSWM 协议帧数；默认 33（必须满足 4n+1）")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="[P0#9] 训练样本硬上限（成本闸门）。None=不限。"
                        "确定性截断划分后的前 N 个，同 (split_seed,N) 可复现。")
    p.add_argument("--max_val_samples", type=int, default=None,
                   help="验证样本硬上限；仅建议 real-model smoke 使用，None=不限。")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader worker 数；profile 默认，A800=0 以避开拥挤 /dev/shm")
    p.add_argument("--prefetch_factor", type=int, default=None,
                   help="仅 num_workers>0 生效；A800 profile 默认禁用")
    p.add_argument("--persistent_workers", action="store_true", default=False,
                   help="保持 DataLoader worker；共享 A800 默认关闭")

    p.add_argument("--device_id", type=int, default=None,
                   help="逻辑 CUDA 索引；默认使用 CUDA_VISIBLE_DEVICES 内的当前设备")
    p.add_argument(
        "--hardware_profile",
        type=str,
        choices=["auto", "a800_80g", "h200_141g"],
        default=os.environ.get("RSWM_HARDWARE_PROFILE", "auto"),
        help="硬件安全配置；A800 80GB 使用 a800_80g（bf16、auto offload、torch.AdamW）",
    )
    p.add_argument(
        "--offload",
        type=str,
        choices=["auto", "0", "1"],
        default=None,
        help=("专家驻留策略；默认读取 OFFLOAD，未设置时为 auto。"
              "auto 按当前可见逻辑 GPU 的总/空闲显存选择；"
              "1=单专家 CPU/GPU swap，0=双专家常驻（A800 上仅用于显式吞吐测试）"),
    )
    p.add_argument("--t5_cpu", action="store_true", help="T5 放 CPU")
    p.add_argument("--vae_cpu", action="store_true", help="VAE 放 CPU")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=None,
                   help="train/val 划分种子;默认=--seed(向后兼容)。多训练种子实验必须固定"
                        "(如 --seed 43 --split_seed 42),否则划分漂移+共享验证集泄漏进训练")
    p.add_argument("--split_file", type=str, default=None,
                   help="预注册 train/val 或官方三路 split JSON；给定时不随机划分，"
                        "且 official test 永不进入训练器")
    # [v2.8 ④] 位姿通路三态 —— 三个臂的唯一区分点,必须写进 checkpoint 与
    # 评估元数据;与 H5 的 pose_injection 冲突时硬错(见 _assert_pose_mode)
    p.add_argument("--pose_mode", type=str, default="native",
                   choices=["native", "identity", "drop"],
                   help="native=真实轨迹 Plücker;identity=静止相机诊断臂"
                        "(恒等位姿仍走 Plücker 通路,不是 text-only);"
                        "drop=不传 dit_cond_dict(真正 no-pose/text-only)")
    p.add_argument("--lora_targets", type=str, default="all",
                   choices=["all", "attn_cam", "attn_only"],
                   help="LoRA 目标模块集 (advisor point 5, 参数效率): "
                        "all=全部 block Linear(注意力+相机注入+FFN); "
                        "attn_cam=仅注意力投影+相机注入(去 FFN); "
                        "attn_only=仅注意力投影")

    p.add_argument("--low_mem", action="store_true", default=True,
                   help="省显存预设：默认 ON（A800/A100 80GB）；传 --no_low_mem 关闭")
    p.add_argument("--no_low_mem", dest="low_mem", action="store_false")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "no"],
        default="bf16",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=None,
                   help="默认由 hardware profile 决定；A800/H200 正式 profile 均为 4")
    p.add_argument("--attention_backend", type=str, default=None,
                   choices=["auto", "flash_attn", "sdpa"],
                   help="auto=A800 优先 FA2、Hopper 可用 FA3，失败回退 SDPA")

    p.add_argument("--use_8bit_adam", action="store_true", default=False,
                   help="使用 bitsandbytes AdamW8bit；仅适合专家常驻 GPU 的大显存配置")
    p.add_argument("--no_8bit_adam", dest="use_8bit_adam", action="store_false")
    p.add_argument(
        "--inactive_model_cpu",
        action="store_true",
        default=None,
        help="兼容旧入口：等价 --offload 1",
    )
    p.add_argument(
        "--no_inactive_model_cpu",
        dest="inactive_model_cpu",
        action="store_false",
        help="兼容旧入口：等价 --offload 0",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    try:
        args = _apply_hardware_profile(parse_args())

        if os.environ.get("RANK", "0") == "0":
            logger.info(
                "训练配置: hardware_profile=%s low_mem=%s bf16=%s "
                "batch=%d accum=%d offload=%s->%s inactive_model_cpu=%s optimizer=%s "
                "attention=%s workers=%d prefetch=%s persistent=%s",
                args.hardware_profile, args.low_mem,
                args.mixed_precision == "bf16", args.batch_size,
                args.gradient_accumulation_steps, args.offload_requested,
                args.offload_resolved, args.inactive_model_cpu,
                "AdamW8bit" if args.use_8bit_adam else "torch.AdamW",
                args.attention_backend, args.num_workers,
                args.prefetch_factor, args.persistent_workers)

        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )

        ckpt = Path(args.checkpoint_dir)
        if not ckpt.is_dir():
            raise FileNotFoundError(f"checkpoint_dir 不存在: {args.checkpoint_dir}")
        data_path = Path(args.data_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"data_path 不存在: {args.data_path}")

        distributed_keys = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
        present_distributed_keys = distributed_keys.intersection(os.environ)
        if present_distributed_keys and present_distributed_keys != distributed_keys:
            missing = sorted(distributed_keys - present_distributed_keys)
            raise RuntimeError(
                f"分布式环境变量不完整，缺 {missing}；请用 torchrun/accelerate 启动，"
                "不要手工只设置其中一部分。")
        if "RANK" in os.environ:
            init_distributed_group()
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)

        args.runtime_gpu_info = _validate_runtime_gpu(args)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        trainer = RemoteSenseLoRATrainer(args)
        trainer.train()

        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        rank = int(os.environ.get("RANK", 0))
        sys.stderr.write(f"\n[rank {rank}] train.py 异常:\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
