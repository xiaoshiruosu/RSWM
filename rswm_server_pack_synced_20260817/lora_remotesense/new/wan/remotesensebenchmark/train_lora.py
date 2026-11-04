#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Wan2.2 with Multi-Direction Benchmark

This script fine-tunes Wan2.2 video generation models using LoRA
on the Multi-Direction Sliding Window Benchmark dataset.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
import wandb
import json
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.animate.animate_utils import get_loraconfig
from remotesensebenchmark.dataset_loader import MultiDirectionDataset, split_dataset


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_main_process():
    """True if single-GPU or rank 0 in distributed."""
    return not dist.is_initialized() or dist.get_rank() == 0


class LoRATrainer:
    """LoRA trainer for Wan2.2 video generation models (supports DDP)."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        self.is_main = is_main_process()
        
        # Initialize wandb if enabled (only main process)
        if args.use_wandb and self.is_main:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
        
        # Load configuration
        self.config = WAN_CONFIGS[args.model_config]
        
        # Initialize components
        self._init_vae()
        self._init_text_encoder()
        self._init_model()
        self._init_dataset()
        self._init_optimizer()
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def _init_vae(self):
        """Initialize VAE"""
        logger.info("Loading VAE...")
        vae_path = os.path.join(self.args.checkpoint_dir, self.config.vae_checkpoint)
        
        if 'Wan2.2' in self.config.vae_checkpoint:
            self.vae = Wan2_2_VAE(vae_pth=vae_path, device=self.device)
        else:
            self.vae = Wan2_1_VAE(vae_pth=vae_path, device=self.device)
        
        self.vae.model.eval().requires_grad_(False)
        logger.info("VAE loaded successfully")
    
    def _init_text_encoder(self):
        """Initialize T5 text encoder"""
        logger.info("Loading T5 text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(self.args.checkpoint_dir, self.config.t5_checkpoint),
            tokenizer_path=os.path.join(self.args.checkpoint_dir, self.config.t5_tokenizer)
        )
        self.text_encoder.model.eval().requires_grad_(False)
        logger.info("T5 encoder loaded successfully")
    
    def _init_model(self):
        """Initialize Wan model with LoRA"""
        logger.info("Loading Wan model...")

        # Load pretrained model in bfloat16
        base_model = WanModel.from_pretrained(
            self.args.checkpoint_dir,
            torch_dtype=self.config.param_dtype
        )

        # Configure LoRA
        logger.info(f"Configuring LoRA with rank={self.args.lora_rank}, alpha={self.args.lora_alpha}")
        lora_config = get_loraconfig(
            transformer=base_model,
            rank=self.args.lora_rank,
            alpha=self.args.lora_alpha,
            init_lora_weights="gaussian"
        )

        # Add LoRA adapters using get_peft_model
        self.model = get_peft_model(base_model, lora_config)

        # Move entire model to device and bfloat16
        # This ensures both base model and LoRA parameters are in the same dtype
        self.model.to(device=self.device, dtype=torch.bfloat16)

        # Wrap with DDP when running distributed
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.args.device_id])

        if self.is_main:
            logger.info(f"Model moved to {self.device} with dtype bfloat16")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({100 * trainable_params / total_params:.2f}%)")
    
    def _init_dataset(self):
        """Initialize dataset and dataloader with train/val split"""
        logger.info(f"Loading dataset from {self.args.data_path}")
        
        # Load full dataset
        full_dataset = MultiDirectionDataset(
            h5_path=self.args.data_path,
            scene_categories=self.args.scene_categories,
            normalize=True,
            load_full_scene=False
        )
        
        # Split into train and validation sets
        train_subset, val_subset = split_dataset(
            full_dataset,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            seed=self.args.seed,
            stratify_by=self.args.stratify_by
        )
        
        self.train_dataset = train_subset
        self.val_dataset = val_subset
        
        # Samplers for distributed training
        train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True
        ) if dist.is_initialized() else None
        val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False
        ) if dist.is_initialized() else None
        
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            drop_last=False
        )
        
        if self.is_main:
            logger.info(f"Dataset loaded: {len(self.train_dataset)} training samples, "
                       f"{len(self.val_dataset)} validation samples")
            logger.info(f"Training batches: {len(self.train_loader)}, "
                       f"Validation batches: {len(self.val_loader)}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )
        
        total_steps = len(self.train_loader) * self.args.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.min_lr
        )
        
        logger.info(f"Optimizer initialized: lr={self.args.learning_rate}, "
                   f"total_steps={total_steps}")

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute training loss for video generation.

        Uses flow matching / diffusion loss with direction conditioning.
        """
        # Get video frames and encode with VAE
        frames = batch['sliding_frames']  # (B, T, C, H, W)
        B, T, C, H, W = frames.shape

        # VAE expects input as (C, T, H, W) for each video in a list
        # Encode each video in the batch
        with torch.no_grad():
            latents_list = []
            for i in range(B):
                # Get single video: (T, C, H, W) -> (C, T, H, W)
                video = frames[i].permute(1, 0, 2, 3)  # (C, T, H, W)
                # VAE encode expects a list of videos, convert to bfloat16 for VAE
                latent = self.vae.encode([video.to(device=self.device, dtype=torch.bfloat16)])[0]  # (Z, t, h, w)
                latents_list.append(latent)

            # Stack latents: (B, Z, t, h, w)
            latents = torch.stack(latents_list, dim=0)
            # Permute to (B, t, Z, h, w) for consistency
            latents = latents.permute(0, 2, 1, 3, 4)  # (B, t, Z, h, w)

        # Ensure latents are bfloat16 for model computation
        latents = latents.to(dtype=torch.bfloat16)

        # Sample timesteps
        timesteps = torch.randint(
            0, self.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # Add noise (flow matching) - keep noise as bfloat16
        noise = torch.randn_like(latents)  # Will be bfloat16 since latents is bfloat16
        t_expanded = timesteps.view(B, 1, 1, 1, 1).to(dtype=torch.bfloat16) / self.config.num_train_timesteps
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise

        # Get direction embeddings
        direction_embeds = batch['segment_embeddings'].to(device=self.device, dtype=torch.bfloat16)  # (B, NumSeg, D)

        # Create text embeddings (use simple prompt for now)
        # In practice, you might want to use more sophisticated prompts
        text_prompts = ["A video showing directional motion"] * B
        with torch.no_grad():
            if not self.args.t5_cpu:
                self.text_encoder.model.to(self.device)
            # Use __call__ instead of encode, T5 already outputs bfloat16
            text_embeds = self.text_encoder(text_prompts, self.device)  # List of [L, D]
            # Ensure text embeddings are bfloat16
            text_embeds = [t.to(dtype=torch.bfloat16) for t in text_embeds]
            if not self.args.t5_cpu:
                self.text_encoder.model.cpu()

        # Prepare latents as list (model expects list of tensors)
        # Convert from (B, t, Z, h, w) to list of (Z, t, h, w)
        # Ensure all tensors are bfloat16
        noisy_x_list = [noisy_latents[i].permute(1, 0, 2, 3).to(dtype=torch.bfloat16) for i in range(B)]

        # Get sequence length (max number of tokens)
        seq_len = self.config.text_len

        # Forward pass through model with autocast for consistent dtype handling
        # WanModel.forward expects: x (list), t (tensor), context (list), seq_len (int), y (list, optional)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model_output_list = self.model(
                x=noisy_x_list,
                t=timesteps,
                context=text_embeds,
                seq_len=seq_len,
                y=None  # No conditional input for T2V
            )

        # Convert output back to tensor
        # model_output_list is a list of (Z, t, h, w)
        model_output = torch.stack([out.permute(1, 0, 2, 3) for out in model_output_list], dim=0)  # (B, t, Z, h, w)

        # Compute loss (predict velocity for flow matching)
        # Convert to float32 for loss computation for numerical stability
        target = (noise - latents).float()
        model_output = model_output.float()
        loss = F.mse_loss(model_output, target, reduction='mean')

        return loss

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.num_epochs}", disable=not self.is_main)

        for step, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            # Forward pass
            loss = self.compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

            if self.args.use_wandb and self.is_main and step % self.args.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': epoch * len(self.train_loader) + step
                })

        return {'loss': total_loss / len(self.train_loader)}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation {epoch}", disable=not self.is_main)
            
            for step, batch in enumerate(pbar):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Forward pass
                loss = self.compute_loss(batch)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                avg_loss = total_loss / num_batches
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': f'{loss.item():.4f}',
                    'avg_val_loss': f'{avg_loss:.4f}'
                })
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        if dist.is_initialized():
            avg_val_t = torch.tensor([avg_val_loss], device=self.device)
            dist.all_reduce(avg_val_t, op=dist.ReduceOp.AVG)
            avg_val_loss = avg_val_t.item()
        
        if self.args.use_wandb and self.is_main:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/epoch': epoch
            })
        
        return {'loss': avg_val_loss}

    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Save model checkpoint with complete training parameters (rank 0 only in DDP)."""
        if dist.is_initialized():
            dist.barrier()
        if not self.is_main:
            return None

        # Create checkpoint directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights (unwrap DDP if needed)
        model_to_save = self.model.module if dist.is_initialized() else self.model
        model_to_save.save_pretrained(save_dir)
        logger.info(f"LoRA weights saved to {save_dir}")

        # Save training state (optimizer, scheduler, etc.)
        state = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model_config': self.args.model_config,
            'lora_rank': self.args.lora_rank,
            'lora_alpha': self.args.lora_alpha,
        }
        torch.save(state, save_dir / 'training_state.pt')
        logger.info(f"Training state saved to {save_dir / 'training_state.pt'}")

        # Save complete training configuration as JSON
        config_dict = {
            'timestamp': timestamp,
            'epoch': epoch,
            'train_metrics': {k: float(v) for k, v in train_metrics.items()},
            'val_metrics': {k: float(v) for k, v in val_metrics.items()},
            'model_config': self.args.model_config,
            'checkpoint_dir': self.args.checkpoint_dir,
            'data_path': self.args.data_path,
            'lora_config': {
                'rank': self.args.lora_rank,
                'alpha': self.args.lora_alpha,
            },
            'training_hyperparameters': {
                'batch_size': self.args.batch_size,
                'num_epochs': self.args.num_epochs,
                'learning_rate': self.args.learning_rate,
                'min_lr': self.args.min_lr,
                'weight_decay': self.args.weight_decay,
                'adam_beta1': self.args.adam_beta1,
                'adam_beta2': self.args.adam_beta2,
                'max_grad_norm': self.args.max_grad_norm,
            },
            'data_config': {
                'train_ratio': self.args.train_ratio,
                'val_ratio': self.args.val_ratio,
                'stratify_by': self.args.stratify_by,
                'scene_categories': self.args.scene_categories,
                'num_workers': self.args.num_workers,
                'pin_memory': self.args.pin_memory,
                'train_samples': len(self.train_dataset),
                'val_samples': len(self.val_dataset),
            },
            'device_config': {
                'device_id': self.args.device_id,
                't5_cpu': self.args.t5_cpu,
            },
            'seed': self.args.seed,
            'wandb_config': {
                'use_wandb': self.args.use_wandb,
                'wandb_project': self.args.wandb_project,
                'wandb_run_name': self.args.wandb_run_name,
            } if self.args.use_wandb else None,
        }
        
        with open(save_dir / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Training configuration saved to {save_dir / 'training_config.json'}")

        # Save a copy of the original config file if it exists
        config_file = Path(self.args.data_path).parent / 'train_config.yaml'
        if config_file.exists():
            shutil.copy2(config_file, save_dir / 'original_train_config.yaml')
            logger.info(f"Original config file copied to {save_dir / 'original_train_config.yaml'}")

        logger.info(f"Checkpoint saved to {save_dir}")
        return save_dir

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        # Track best validation loss
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(1, self.args.num_epochs + 1):
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate(epoch)

            if self.is_main:
                logger.info(f"Epoch {epoch}/{self.args.num_epochs} - "
                           f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Val Loss: {val_metrics['loss']:.4f}")

            # Save checkpoint (rank 0 only in DDP)
            checkpoint_dir = self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            if self.is_main and checkpoint_dir is not None:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_epoch = epoch
                    best_dir = Path(self.args.output_dir) / "best_checkpoint"
                    if best_dir.exists():
                        if best_dir.is_symlink():
                            best_dir.unlink()
                        else:
                            shutil.rmtree(best_dir)
                    shutil.copytree(checkpoint_dir, best_dir)
                    logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")

        if self.is_main:
            summary = {
                'total_epochs': self.args.num_epochs,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'total_training_time': time.time() - start_time,
                'final_train_loss': train_metrics['loss'],
                'final_val_loss': val_metrics['loss'],
            }
            summary_path = Path(self.args.output_dir) / 'training_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Training summary saved to {summary_path}")

        total_time = time.time() - start_time
        if self.is_main:
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Best model: epoch {best_epoch} with val_loss {best_val_loss:.4f}")
            if self.args.use_wandb:
                wandb.finish()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning for Wan2.2 with Multi-Direction Benchmark"
    )

    # Model and checkpoint paths
    parser.add_argument(
        '--model_config',
        type=str,
        default='ti2v-5B',
        choices=['t2v-A14B', 'i2v-A14B', 'ti2v-5B', 'animate-14B', 's2v-14B'],
        help='Model configuration to use'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to pretrained model checkpoint directory'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to Multi-Direction Benchmark HDF5 file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./lora_checkpoints',
        help='Directory to save LoRA checkpoints'
    )

    # LoRA configuration
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=128,
        help='LoRA rank (r parameter)'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=128,
        help='LoRA alpha parameter'
    )

    # Training hyperparameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Training batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-6,
        help='Minimum learning rate for cosine scheduler'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay'
    )
    parser.add_argument(
        '--adam_beta1',
        type=float,
        default=0.9,
        help='Adam beta1'
    )
    parser.add_argument(
        '--adam_beta2',
        type=float,
        default=0.999,
        help='Adam beta2'
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (0 to disable)'
    )

    # Data configuration
    parser.add_argument(
        '--scene_categories',
        type=str,
        nargs='+',
        default=None,
        help='Scene categories to use (default: all)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of data loading workers (shared-server safe default: 0)'
    )
    parser.add_argument(
        '--pin_memory', action='store_true', default=False,
        help='Pin DataLoader memory (off by default on the shared A800 server)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of training samples (default: 0.8)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
        help='Ratio of validation samples (default: 0.2)'
    )
    parser.add_argument(
        '--stratify_by',
        type=str,
        default=None,
        help='Metadata key to stratify train/val split by (e.g., "scene_category")'
    )

    # Logging and checkpointing
    parser.add_argument(
        '--save_interval',
        type=int,
        default=1,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='Log metrics every N steps'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='wan2.2-lora-finetuning',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='W&B run name'
    )

    # Device configuration
    parser.add_argument(
        '--device_id',
        type=int,
        default=None,
        help='Logical CUDA index (default: LOCAL_RANK/current device)'
    )
    parser.add_argument(
        '--t5_cpu',
        action='store_true',
        help='Keep T5 encoder on CPU to save VRAM'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    # When launched with torchrun, use LOCAL_RANK as device_id
    if "LOCAL_RANK" in os.environ:
        args.device_id = int(os.environ["LOCAL_RANK"])
    elif args.device_id is None and torch.cuda.is_available():
        args.device_id = torch.cuda.current_device()
    return args


def main():
    """Main training function (supports single-GPU and torchrun DDP)."""
    args = parse_args()

    # Initialize distributed training when launched with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory (rank 0 only in DDP)
    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        trainer = LoRATrainer(args)
        trainer.train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
