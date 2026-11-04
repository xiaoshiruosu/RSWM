#!/usr/bin/env python3
"""
Inference Script for Wan2.2 with Trained LoRA Weights

This script demonstrates how to use a fine-tuned LoRA model
for video generation with directional control.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wan.configs import WAN_CONFIGS
from wan.textimage2video import WanTI2V
from wan.text2video import WanT2V
from wan.image2video import WanI2V


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_with_lora(
    model_config: str,
    checkpoint_dir: str,
    lora_checkpoint: str,
    device_id=None,
    lora_scale: float = 0.8,
):
    """
    Load Wan2.2 model with LoRA weights.
    
    Args:
        model_config: Model configuration name (e.g., 'ti2v-5B')
        checkpoint_dir: Path to base model checkpoint
        lora_checkpoint: Path to LoRA checkpoint
        device_id: logical CUDA index; defaults to LOCAL_RANK/current device
        lora_scale: LoRA weight scale factor (default 0.8, reduces overfitting noise)
    
    Returns:
        Loaded model with LoRA weights
    """
    if device_id is None:
        device_id = int(os.environ.get(
            "LOCAL_RANK",
            torch.cuda.current_device() if torch.cuda.is_available() else 0))
    logger.info(f"Loading model: {model_config} on logical CUDA device {device_id}")
    
    # Get configuration
    cfg = WAN_CONFIGS[model_config]
    
    # Load base model based on type
    if 'ti2v' in model_config.lower():
        model = WanTI2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            convert_model_dtype=True,
            t5_cpu=True
        )
    elif 't2v' in model_config.lower():
        model = WanT2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            convert_model_dtype=True
        )
    elif 'i2v' in model_config.lower():
        model = WanI2V(
            config=cfg,
            checkpoint_dir=checkpoint_dir,
            device_id=device_id,
            convert_model_dtype=True
        )
    else:
        raise ValueError(f"Unsupported model config: {model_config}")
    
    # Load LoRA weights.
    # Handle both file path and directory path
    # If it's a file path (ends with adapter_model.safetensors or adapter_model.bin),
    # extract the directory path
    lora_checkpoint_path = Path(lora_checkpoint)
    lora_path = lora_checkpoint
    
    # Check if it's a file
    if lora_checkpoint_path.is_file():
        # Extract directory from file path
        lora_path = str(lora_checkpoint_path.parent)
        logger.info(f"Detected file path '{lora_checkpoint}', using directory: {lora_path}")
    elif lora_checkpoint_path.is_dir():
        # Already a directory, use as is
        lora_path = str(lora_checkpoint_path)
        logger.info(f"Using directory path: {lora_path}")
    else:
        # Try to resolve the path
        resolved_path = lora_checkpoint_path.resolve()
        if resolved_path.is_file():
            lora_path = str(resolved_path.parent)
            logger.info(f"Resolved to file path, using directory: {lora_path}")
        elif resolved_path.is_dir():
            lora_path = str(resolved_path)
            logger.info(f"Resolved to directory path: {lora_path}")
        else:
            raise ValueError(
                f"LoRA checkpoint path does not exist: {lora_checkpoint}\n"
                f"Resolved path: {resolved_path}"
            )
    
    # Verify the directory contains necessary files
    lora_dir = Path(lora_path)
    adapter_config_path = lora_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        # List files in directory for debugging
        files_in_dir = list(lora_dir.iterdir()) if lora_dir.exists() else []
        raise ValueError(
            f"LoRA directory '{lora_path}' does not contain 'adapter_config.json'.\n"
            f"Files found in directory: {[f.name for f in files_in_dir]}\n"
            f"Please ensure the LoRA checkpoint directory contains both 'adapter_config.json' and 'adapter_model.safetensors' (or 'adapter_model.bin')"
        )
    
    logger.info(f"Loading LoRA weights from: {lora_path}")
    model.model = PeftModel.from_pretrained(
        model.model,
        lora_path
    )

    # LoRA weight scaling: scale down to reduce overfitting-induced noise
    if lora_scale != 1.0:
        num_scaled = 0
        for name, module in model.model.named_modules():
            if "lora_" not in name:
                continue
            if not hasattr(module, "scaling"):
                continue
            scaling = module.scaling
            if isinstance(scaling, dict):
                if "default" in scaling:
                    module.scaling["default"] = module.scaling["default"] * lora_scale
                    num_scaled += 1
            elif isinstance(scaling, (int, float)):
                module.scaling = scaling * lora_scale
                num_scaled += 1
        logger.info(
            f"LoRA weight scaling applied: scale={lora_scale}, "
            f"modules adjusted={num_scaled}"
        )
    else:
        logger.info("LoRA weight scaling skipped (lora_scale=1.0)")

    logger.info("Model loaded successfully with LoRA weights")
    return model


def generate_video(
    model,
    prompt: str,
    image_path: str = None,
    size: tuple = (1280, 704),
    frame_num: int = 33,
    output_path: str = "output_lora.mp4",
    seed: int = 42
):
    """
    Generate video using the LoRA-enhanced model.
    
    Args:
        model: Loaded model with LoRA
        prompt: Text prompt for generation
        image_path: Optional image path for I2V
        size: Video size (width, height)
        frame_num: Number of frames
        output_path: Output video path
        seed: Random seed
    """
    logger.info(f"Generating video with prompt: {prompt}")
    
    # Generate video
    if hasattr(model, 'generate'):
        if image_path:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            # For TI2V, first parameter is input_prompt (positional), then keyword arguments
            if 'ti2v' in model.__class__.__name__.lower():
                video = model.generate(
                    prompt,  # positional argument for input_prompt
                    img=img,
                    size=size,
                    max_area=size[0] * size[1],  # max_area is required for TI2V
                    frame_num=frame_num,
                    seed=seed,
                    offload_model=True
                )
            else:
                video = model.generate(
                    prompt,  # positional argument for input_prompt
                    img=img,
                    size=size,
                    frame_num=frame_num,
                    seed=seed,
                    offload_model=True
                )
        else:
            # For text-to-video, first parameter is input_prompt (positional)
            video = model.generate(
                prompt,  # positional argument for input_prompt
                size=size,
                frame_num=frame_num,
                seed=seed,
                offload_model=True
            )
    else:
        raise ValueError("Model does not have generate method")
    
    # Save video
    logger.info(f"Saving video to: {output_path}")
    logger.info(f"Video tensor shape: {video.shape}, dtype: {video.dtype}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory path (not just filename)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created/verified output directory: {output_dir}")
    else:
        # If no directory, save to current working directory
        output_path = os.path.join(os.getcwd(), os.path.basename(output_path))
        logger.info(f"No directory specified, using current directory: {os.getcwd()}")
        logger.info(f"Updated output path: {output_path}")
    
    # Import required modules
    from wan.configs import WAN_CONFIGS
    import imageio
    import torchvision
    
    # Get FPS from config (default to 8 if not available)
    try:
        # Try to get model config name from model class
        model_class_name = model.__class__.__name__
        if 'TI2V' in model_class_name:
            config_key = 'ti2v-5B'  # Default, could be improved
        elif 'T2V' in model_class_name:
            config_key = 't2v-A14B'  # Default
        elif 'I2V' in model_class_name:
            config_key = 'i2v-A14B'  # Default
        else:
            config_key = None
        
        if config_key and config_key in WAN_CONFIGS:
            cfg = WAN_CONFIGS[config_key]
            fps = cfg.sample_fps if hasattr(cfg, 'sample_fps') else 8
        else:
            fps = 8
        logger.info(f"Using FPS: {fps}")
    except Exception as e:
        logger.warning(f"Could not get FPS from config: {e}, using default 8")
        fps = 8
    
    # Save video with proper error handling and verification
    # Video tensor shape: (C, N, H, W) where C=3, N=frame_num
    logger.info(f"Attempting to save video to: {output_path}")
    logger.info(f"Video tensor shape: {video.shape}, dtype: {video.dtype}, device: {video.device}")
    
    try:
        # Ensure video is on CPU and in correct format
        if video.is_cuda:
            video = video.cpu()
            logger.info("Moved video tensor to CPU")
        
        # Process video tensor for saving
        # Clamp values to range [-1, 1]
        video_processed = video.clamp(min=-1.0, max=1.0)
        
        # Convert to grid format: (N, H, W, C) where N is number of frames
        # Unbind along frame dimension (dim=1 for (C, N, H, W))
        frames = []
        for frame_idx in range(video_processed.shape[1]):  # Iterate over frames
            frame = video_processed[:, frame_idx, :, :]  # (C, H, W)
            # Make grid (for nrow=1, this just arranges the frame)
            grid_frame = torchvision.utils.make_grid(
                frame.unsqueeze(0),  # Add batch dim: (1, C, H, W)
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            frames.append(grid_frame)
        
        # Stack frames: (N, C, H, W) -> permute to (N, H, W, C)
        video_tensor = torch.stack(frames, dim=0).permute(0, 2, 3, 1)
        video_tensor = (video_tensor * 255).type(torch.uint8).cpu().numpy()
        
        logger.info(f"Processed video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        
        # Write video using imageio with context manager to ensure proper cleanup
        logger.info(f"Opening video writer for: {output_path}")
        with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
            for frame_idx, frame in enumerate(video_tensor):
                writer.append_data(frame)
                if (frame_idx + 1) % 10 == 0:
                    logger.info(f"Written {frame_idx + 1}/{len(video_tensor)} frames")
        
        logger.info(f"Video writer closed. Verifying file existence...")
        
        # Verify file was actually created
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Video file was not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise ValueError(f"Video file was created but is empty (0 bytes) at {output_path}")
        
        logger.info(f"✓ Video generation completed! File saved successfully.")
        logger.info(f"✓ File location: {output_path}")
        logger.info(f"✓ File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
    except Exception as e:
        logger.error(f"✗ Failed to save video: {e}")
        logger.error(f"✗ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"✗ Traceback:\n{traceback.format_exc()}")
        # Check if partial file exists and should be cleaned up
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Removed incomplete file: {output_path}")
            except:
                pass
        raise
    
    return video


def main():
    parser = argparse.ArgumentParser(
        description="Inference with LoRA-enhanced Wan2.2 model"
    )
    
    parser.add_argument(
        '--model_config',
        type=str,
        default='ti2v-5B',
        help='Model configuration'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        required=True,
        help='Path to base model checkpoint'
    )
    parser.add_argument(
        '--lora_checkpoint',
        type=str,
        required=True,
        help='Path to LoRA checkpoint'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        required=True,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Optional image path for I2V'
    )
    parser.add_argument(
        '--size',
        type=str,
        default='1280*704',
        help='Video size (e.g., 1280*704)'
    )
    parser.add_argument(
        '--frame_num',
        type=int,
        default=33,
        help='Number of frames'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_lora.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=None,
        help='Logical CUDA index (default: LOCAL_RANK/current device)'
    )
    parser.add_argument(
        '--lora_scale',
        type=float,
        default=0.8,
        help='LoRA weight scale factor (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    # Force absolute path for output and log it
    args.output = os.path.abspath(args.output)
    logger.info(f"Output file will be saved to absolute path: {args.output}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Parse size
    width, height = map(int, args.size.split('*'))
    size = (width, height)
    
    # Load model with LoRA
    model = load_model_with_lora(
        model_config=args.model_config,
        checkpoint_dir=args.checkpoint_dir,
        lora_checkpoint=args.lora_checkpoint,
        device_id=args.device_id,
        lora_scale=args.lora_scale,
    )
    
    # Generate video
    video = generate_video(
        model=model,
        prompt=args.prompt,
        image_path=args.image,
        size=size,
        frame_num=args.frame_num,
        output_path=args.output,
        seed=args.seed
    )
    
    logger.info("Inference completed successfully!")


if __name__ == '__main__':
    main()
