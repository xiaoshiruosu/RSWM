#!/usr/bin/env python3
"""
Sliding Window GIF Visualizer

This script visualizes the first sample from the sliding window benchmark dataset
as an animated GIF. It shows:
1. The sliding window moving from center to target position
2. The full scene with the current window position highlighted
3. Direction information overlay

Usage:
    python visualize_sliding_window_gif.py
    python visualize_sliding_window_gif.py --input sliding_window_benchmark.h5 --output sliding_window_demo.gif
"""

import os
import sys
import argparse
import numpy as np
import h5py

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available. Install with: pip install imageio")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Default file paths
DEFAULT_INPUT = "sliding_window_benchmark.h5"
DEFAULT_OUTPUT = "sliding_window_demo.gif"


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 (0-255)"""
    if arr.dtype == np.uint8:
        return arr
    
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    else:
        return arr.clip(0, 255).astype(np.uint8)


def draw_rectangle(image: np.ndarray, bounds: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw a rectangle on the image"""
    h_start, h_end, w_start, w_end = bounds
    img = image.copy()
    
    if CV2_AVAILABLE:
        cv2.rectangle(img, (w_start, h_start), (w_end, h_end), color, thickness)
    else:
        # Simple rectangle drawing without cv2
        img[h_start:h_start+thickness, w_start:w_end] = color
        img[h_end-thickness:h_end, w_start:w_end] = color
        img[h_start:h_end, w_start:w_start+thickness] = color
        img[h_start:h_end, w_end-thickness:w_end] = color
    
    return img


def add_text_to_image(image: np.ndarray, text: str, position: tuple, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Add text to image using PIL or fallback"""
    if PIL_AVAILABLE:
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add background for text
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0))
        draw.text(position, text, fill=color, font=font)
        
        return np.array(pil_img)
    elif CV2_AVAILABLE:
        img = image.copy()
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img
    else:
        return image


def create_sliding_window_gif(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    sample_idx: int = 0,
    fps: int = 5,
    show_full_scene: bool = True
) -> bool:
    """
    Create an animated GIF showing the sliding window movement.
    
    Args:
        input_path: Path to the HDF5 benchmark file
        output_path: Path for output GIF
        sample_idx: Index of sample to visualize
        fps: Frames per second for GIF
        show_full_scene: Whether to show full scene with window overlay
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required. Install with: pip install imageio")
        return False
    
    print(f"Reading from: {input_path}")
    
    try:
        with h5py.File(input_path, 'r') as f:
            # Get sample keys
            sample_keys = sorted(list(f['data'].keys()))
            
            if not sample_keys:
                print("Error: No samples found in dataset")
                return False
            
            if sample_idx >= len(sample_keys):
                print(f"Warning: sample_idx {sample_idx} out of range, using 0")
                sample_idx = 0
            
            sample_name = sample_keys[sample_idx]
            print(f"Processing sample: {sample_name}")
            
            # Load data
            sample = f[f'data/{sample_name}']
            sliding_frames = sample['sliding_frames'][:]
            full_scene = sample['full_scene'][:]
            trajectory = sample['trajectory'][:]
            
            # Load attributes
            direction_angle = sample.attrs['direction_angle']
            overlap_ratio = sample.attrs['overlap_ratio']
            scene_category = sample.attrs['scene_category']
            center_bounds = sample.attrs['center_bounds']
            target_bounds = sample.attrs['target_bounds']
            
            print(f"  Scene: {scene_category}")
            print(f"  Direction: {direction_angle:.1f} degrees")
            print(f"  Overlap: {overlap_ratio:.2%}")
            print(f"  Sliding frames: {sliding_frames.shape}")
            print(f"  Full scene: {full_scene.shape}")
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return False
    
    # Normalize to uint8
    sliding_frames = normalize_to_uint8(sliding_frames)
    full_scene = normalize_to_uint8(full_scene)
    
    num_frames = len(sliding_frames)
    win_h, win_w = sliding_frames.shape[1:3]
    scene_h, scene_w = full_scene.shape[:2]
    
    print(f"\nGenerating GIF with {num_frames} frames...")
    
    frames_for_gif = []
    
    for i in range(num_frames):
        frame = sliding_frames[i]
        cy, cx = trajectory[i]
        
        # Calculate current window bounds
        half_h = win_h // 2
        half_w = win_w // 2
        current_bounds = (
            max(0, int(cy) - half_h),
            min(scene_h, int(cy) + half_h),
            max(0, int(cx) - half_w),
            min(scene_w, int(cx) + half_w)
        )
        
        if show_full_scene:
            # Create side-by-side view: [Full Scene with overlay] | [Window View]
            
            # Draw on full scene
            scene_with_overlay = full_scene.copy()
            
            # Draw center bounds (green)
            scene_with_overlay = draw_rectangle(
                scene_with_overlay, center_bounds, (0, 255, 0), thickness=3
            )
            
            # Draw target bounds (red)
            scene_with_overlay = draw_rectangle(
                scene_with_overlay, target_bounds, (255, 0, 0), thickness=3
            )
            
            # Draw current window (yellow)
            scene_with_overlay = draw_rectangle(
                scene_with_overlay, current_bounds, (255, 255, 0), thickness=2
            )
            
            # Resize full scene to match window height for side-by-side
            scale = win_h / scene_h
            new_scene_w = int(scene_w * scale)
            
            if CV2_AVAILABLE:
                scene_resized = cv2.resize(scene_with_overlay, (new_scene_w, win_h))
            else:
                # Simple resize
                h_idx = np.linspace(0, scene_h - 1, win_h).astype(int)
                w_idx = np.linspace(0, scene_w - 1, new_scene_w).astype(int)
                scene_resized = scene_with_overlay[np.ix_(h_idx, w_idx)]
            
            # Create separator
            separator = np.ones((win_h, 4, 3), dtype=np.uint8) * 255
            
            # Concatenate: [scene] | [separator] | [window]
            combined = np.concatenate([scene_resized, separator, frame], axis=1)
            
            # Add text overlay
            progress = (i + 1) / num_frames * 100
            text = f"Step {i+1}/{num_frames} | Angle: {direction_angle:.0f} deg | Progress: {progress:.0f}%"
            combined = add_text_to_image(combined, text, (10, 10))
            
            # Add legend
            combined = add_text_to_image(combined, "Green: Start | Red: Target | Yellow: Current", (10, win_h - 20))
            
            frames_for_gif.append(combined)
        else:
            # Just the sliding window frames
            frame_with_text = add_text_to_image(
                frame, 
                f"Frame {i+1}/{num_frames} | Angle: {direction_angle:.0f} deg", 
                (5, 5)
            )
            frames_for_gif.append(frame_with_text)
    
    # Save GIF
    print(f"Saving GIF to: {output_path}")
    
    try:
        imageio.mimsave(output_path, frames_for_gif, fps=fps, loop=0)
        print(f"\nSuccess! GIF saved: {output_path}")
        print(f"  - Frames: {len(frames_for_gif)}")
        print(f"  - Size: {frames_for_gif[0].shape[1]}x{frames_for_gif[0].shape[0]}")
        print(f"  - FPS: {fps}")
        return True
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False


def create_comparison_gif(
    input_path: str = DEFAULT_INPUT,
    output_path: str = "sliding_comparison.gif",
    sample_idx: int = 0,
    fps: int = 8
) -> bool:
    """
    Create a simple comparison GIF showing start vs end frames.
    
    Creates an animation that alternates between:
    - Center frame (starting position)
    - Target frame (ending position)
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required")
        return False
    
    print(f"Creating comparison GIF from: {input_path}")
    
    try:
        with h5py.File(input_path, 'r') as f:
            sample_keys = sorted(list(f['data'].keys()))
            
            if not sample_keys:
                print("Error: No samples found")
                return False
            
            sample_name = sample_keys[sample_idx]
            sample = f[f'data/{sample_name}']
            
            center_frame = sample['center_frame'][:]
            target_frame = sample['target_frame'][:]
            direction_angle = sample.attrs['direction_angle']
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    center_frame = normalize_to_uint8(center_frame)
    target_frame = normalize_to_uint8(target_frame)
    
    H, W = center_frame.shape[:2]
    separator = np.ones((H, 4, 3), dtype=np.uint8) * 255
    
    # Add labels
    center_labeled = add_text_to_image(center_frame.copy(), "CENTER (Start)", (5, 5), (0, 255, 0))
    target_labeled = add_text_to_image(target_frame.copy(), "TARGET (End)", (5, 5), (255, 0, 0))
    
    # Side by side
    side_by_side = np.concatenate([center_labeled, separator, target_labeled], axis=1)
    side_by_side = add_text_to_image(side_by_side, f"Direction: {direction_angle:.0f} deg", (5, H - 20))
    
    # Create alternating animation
    frames = [center_labeled, target_labeled] * 4  # 4 cycles
    
    try:
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Comparison GIF saved: {output_path}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Visualize Sliding Window Benchmark as GIF")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input HDF5 file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output GIF file")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument("--simple", action="store_true", help="Create simple window-only GIF")
    parser.add_argument("--comparison", action="store_true", help="Create start/end comparison GIF")
    args = parser.parse_args()
    
    if args.comparison:
        success = create_comparison_gif(
            input_path=args.input,
            output_path=args.output.replace(".gif", "_comparison.gif"),
            sample_idx=args.sample,
            fps=args.fps
        )
    else:
        success = create_sliding_window_gif(
            input_path=args.input,
            output_path=args.output,
            sample_idx=args.sample,
            fps=args.fps,
            show_full_scene=not args.simple
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
