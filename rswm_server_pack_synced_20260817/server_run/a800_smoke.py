#!/usr/bin/env python3
"""Three-level A800 smoke test; no real weights or datasets are downloaded."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "lora_remotesense" / "new"
for path in (str(NEW), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def level1():
    modules = [
        "torch", "torchvision", "flash_attn", "transformers", "tokenizers",
        "diffusers", "accelerate", "peft", "numpy", "h5py",
        "wan.configs", "wan.modules.attention",
        "wan.modules.animate.face_blocks", "wan.modules.s2v.auxi_blocks",
        "dataset_loader", "train",
    ]
    for name in modules:
        importlib.import_module(name)
        print(f"PASS import {name}")


def tiny_lora_model(device="cpu", dtype=None):
    import torch
    from peft import LoraConfig, get_peft_model

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32)

        def forward(self, x):
            return self.proj(x)

    cfg = LoraConfig(r=4, lora_alpha=4, target_modules=["proj"], lora_dropout=0.0)
    model = get_peft_model(Tiny(), cfg).to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise AssertionError("LoRA injection produced zero trainable parameters")
    return model, trainable


def level2():
    import torch
    model, trainable = tiny_lora_model()
    x = torch.randn(2, 32)
    y = model(x)
    if y.shape != x.shape:
        raise AssertionError(f"unexpected shape: {y.shape}")
    print(f"PASS PEFT LoRA injection + forward (trainable={trainable})")

    # Exercise the public function actually imported by WanModel (not only an
    # outer convenience dispatcher), including variable-length fallback.
    previous = os.environ.get("RSWM_ATTENTION_BACKEND")
    os.environ["RSWM_ATTENTION_BACKEND"] = "sdpa"
    try:
        from wan.modules.attention import flash_attention
        q = torch.randn(2, 5, 2, 4)
        k = torch.randn(2, 6, 2, 4)
        v = torch.randn(2, 6, 2, 4)
        out = flash_attention(
            q, k, v, q_lens=torch.tensor([3, 5]),
            k_lens=torch.tensor([4, 6]), dtype=torch.float32)
        if out.shape != q.shape or torch.count_nonzero(out[0, 3:]):
            raise AssertionError("main SDPA variable-length fallback is invalid")
        from wan.modules.animate.face_blocks import attention as face_attention
        from wan.modules.s2v.auxi_blocks import attention as aux_attention
        for name, func in (("face", face_attention), ("s2v", aux_attention)):
            aux_out = func(q, k, v, max_seqlen_q=5, batch_size=2)
            if aux_out.shape != (2, 5, 8) or not torch.isfinite(aux_out).all():
                raise AssertionError(f"{name} SDPA fallback is invalid")
    finally:
        if previous is None:
            os.environ.pop("RSWM_ATTENTION_BACKEND", None)
        else:
            os.environ["RSWM_ATTENTION_BACKEND"] = previous
    print("PASS real Wan public/auxiliary attention SDPA fallback")


def level3():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Level 3 requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Level 3 requires exactly one visible GPU. Set CUDA_VISIBLE_DEVICES "
            "to a GPU allocated to you; this test will not choose one.")
    major, minor = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    name = torch.cuda.get_device_properties(0).name
    if "A800" not in name or major != 8 or not (75 <= total < 90):
        raise RuntimeError(
            f"Expected one NVIDIA A800 80GB(sm80), got {name}, "
            f"sm={major}.{minor}, {total:.1f}GiB")

    torch.cuda.reset_peak_memory_stats()
    model, trainable = tiny_lora_model("cuda", torch.bfloat16)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    target = torch.zeros_like(x)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out.float(), target.float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Exercise the exact helper used by 14B expert swapping: optimizer state
    # must follow LoRA Parameters CPU -> GPU, and gradients must survive a
    # round-trip before the next AdamW update.
    from train import RemoteSenseLoRATrainer

    class OptimizerOwner:
        pass

    owner = OptimizerOwner()
    owner.optimizer = optimizer
    model.cpu()
    RemoteSenseLoRATrainer._move_optimizer_state(owner, model, torch.device("cpu"))
    if any(value.device.type != "cpu" for state in optimizer.state.values()
           for value in state.values() if torch.is_tensor(value)):
        raise AssertionError("optimizer state did not move to CPU")
    model.to(device="cuda", dtype=torch.bfloat16)
    RemoteSenseLoRATrainer._move_optimizer_state(owner, model, torch.device("cuda"))
    if any(value.device.type != "cuda" for state in optimizer.state.values()
           for value in state.values() if torch.is_tensor(value)):
        raise AssertionError("optimizer state did not return to CUDA")
    loss2 = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss2.backward()
    model.cpu()
    RemoteSenseLoRATrainer._move_optimizer_state(owner, model, torch.device("cpu"))
    model.to(device="cuda", dtype=torch.bfloat16)
    RemoteSenseLoRATrainer._move_optimizer_state(owner, model, torch.device("cuda"))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Enter the real RSWM attention dispatcher: A800 should select FA2; ABI or
    # kernel failures must fall back to SDPA rather than aborting the process.
    from wan.modules.attention import attention, attention_backend_status
    q = torch.randn(1, 64, 4, 32, device="cuda", dtype=torch.bfloat16)
    attn_out = attention(q, q, q, dtype=torch.bfloat16)
    if attn_out.shape != q.shape or not torch.isfinite(attn_out).all():
        raise AssertionError("attention smoke produced invalid output")
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    print(f"PASS one BF16 LoRA forward/backward/AdamW step (trainable={trainable})")
    print("PASS LoRA gradient + torch.AdamW state CPU/CUDA expert-swap round trip")
    print(f"PASS RSWM attention dispatcher {attention_backend_status()} peak={peak:.3f}GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, choices=(1, 2, 3), default=3)
    args = ap.parse_args()
    os.environ.setdefault("RSWM_ATTENTION_BACKEND", "auto")
    level1()
    if args.level >= 2:
        level2()
    if args.level >= 3:
        level3()
    print(f"A800 smoke level {args.level}: PASS")


if __name__ == "__main__":
    main()
