#!/usr/bin/env python3
"""
Multi-Direction Sliding Window GIF Visualizer

Visualizes the multi-directional sliding window path from the benchmark dataset.
Shows the window following a path with multiple direction changes.

Features:
- Full scene view with complete trajectory path
- Current window position with direction indicator
- Segment-by-segment color coding
- Direction angle annotations
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

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


DEFAULT_INPUT = "multi_direction_benchmark.h5"
DEFAULT_OUTPUT = "multi_direction_demo.gif"

# Segment colors (RGB)
SEGMENT_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
]


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 (0-255)"""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.clip(0, 255).astype(np.uint8)


def draw_line(image: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw a line on the image"""
    img = image.copy()
    if CV2_AVAILABLE:
        cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), color, thickness)
    else:
        # Simple line using numpy (Bresenham-like)
        y0, x0 = int(pt1[0]), int(pt1[1])
        y1, x1 = int(pt2[0]), int(pt2[1])
        
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y0 < y1 else -1
        sx = 1 if x0 < x1 else -1
        err = dx - dy
        
        while True:
            for t in range(-thickness//2, thickness//2 + 1):
                if 0 <= y0+t < img.shape[0] and 0 <= x0 < img.shape[1]:
                    img[y0+t, x0] = color
                if 0 <= y0 < img.shape[0] and 0 <= x0+t < img.shape[1]:
                    img[y0, x0+t] = color
            
            if y0 == y1 and x0 == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    return img


def draw_rectangle(image: np.ndarray, bounds: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw a rectangle on the image"""
    h_start, h_end, w_start, w_end = bounds
    img = image.copy()
    
    if CV2_AVAILABLE:
        cv2.rectangle(img, (w_start, h_start), (w_end, h_end), color, thickness)
    else:
        img[h_start:h_start+thickness, w_start:w_end] = color
        img[h_end-thickness:h_end, w_start:w_end] = color
        img[h_start:h_end, w_start:w_start+thickness] = color
        img[h_start:h_end, w_end-thickness:w_end] = color
    
    return img


def draw_circle(image: np.ndarray, center: tuple, radius: int, color: tuple, thickness: int = -1) -> np.ndarray:
    """Draw a circle on the image"""
    img = image.copy()
    cy, cx = int(center[0]), int(center[1])
    
    if CV2_AVAILABLE:
        cv2.circle(img, (cx, cy), radius, color, thickness)
    else:
        # Simple filled circle
        for y in range(max(0, cy-radius), min(img.shape[0], cy+radius+1)):
            for x in range(max(0, cx-radius), min(img.shape[1], cx+radius+1)):
                if (y - cy)**2 + (x - cx)**2 <= radius**2:
                    img[y, x] = color
    
    return img


def add_text_to_image(image: np.ndarray, text: str, position: tuple, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Add text to image"""
    if PIL_AVAILABLE:
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=(0, 0, 0))
        draw.text(position, text, fill=color, font=font)
        
        return np.array(pil_img)
    elif CV2_AVAILABLE:
        img = image.copy()
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return img
    return image


def create_multi_direction_gif(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    sample_idx: int = 0,
    fps: int = 5
) -> bool:
    """
    Create an animated GIF showing multi-directional sliding window movement.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required")
        return False
    
    print(f"Reading from: {input_path}")
    
    try:
        with h5py.File(input_path, 'r') as f:
            sample_keys = sorted(list(f['data'].keys()))
            
            if not sample_keys:
                print("Error: No samples found")
                return False
            
            if sample_idx >= len(sample_keys):
                sample_idx = 0
            
            sample_name = sample_keys[sample_idx]
            print(f"Processing sample: {sample_name}")
            
            sample = f[f'data/{sample_name}']
            sliding_frames = sample['sliding_frames'][:]
            full_scene = sample['full_scene'][:]
            waypoints = sample['waypoints'][:]
            segment_angles = sample['segment_angles'][:]
            
            num_segments = sample.attrs['num_segments']
            steps_per_segment = sample.attrs['steps_per_segment']
            scene_category = sample.attrs['scene_category']
            overlap_ratio = sample.attrs['overlap_ratio']
            
            # Get window size from metadata
            window_size = f['metadata'].attrs['window_size']
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    sliding_frames = normalize_to_uint8(sliding_frames)
    full_scene = normalize_to_uint8(full_scene)
    
    print(f"  Scene: {scene_category}")
    print(f"  Segments: {num_segments}")
    print(f"  Angles: {[f'{a:.0f}°' for a in segment_angles]}")
    print(f"  Total frames: {len(sliding_frames)}")
    
    H, W = full_scene.shape[:2]
    win_h, win_w = window_size
    half_h, half_w = win_h // 2, win_w // 2
    
    frames_for_gif = []
    
    for frame_idx in range(len(sliding_frames)):
        frame = sliding_frames[frame_idx]
        
        # Determine current segment
        seg_idx = frame_idx // steps_per_segment
        step_in_seg = frame_idx % steps_per_segment
        
        if seg_idx >= num_segments:
            seg_idx = num_segments - 1
        
        # Get current position by interpolation
        if seg_idx < len(waypoints) - 1:
            start_wp = waypoints[seg_idx]
            end_wp = waypoints[seg_idx + 1]
            t = step_in_seg / (steps_per_segment - 1) if steps_per_segment > 1 else 0
            current_cy = int(start_wp[0] + t * (end_wp[0] - start_wp[0]))
            current_cx = int(start_wp[1] + t * (end_wp[1] - start_wp[1]))
        else:
            current_cy, current_cx = waypoints[-1]
        
        # Create scene with overlay
        scene_overlay = full_scene.copy()
        
        # Draw complete trajectory path
        for i in range(len(waypoints) - 1):
            color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            scene_overlay = draw_line(
                scene_overlay,
                (waypoints[i][0], waypoints[i][1]),
                (waypoints[i+1][0], waypoints[i+1][1]),
                color, thickness=3
            )
        
        # Draw waypoints
        for i, wp in enumerate(waypoints):
            if i == 0:
                color = (0, 255, 0)  # Green for start
            elif i == len(waypoints) - 1:
                color = (255, 0, 0)  # Red for end
            else:
                color = (255, 255, 0)  # Yellow for intermediate
            scene_overlay = draw_circle(scene_overlay, (wp[0], wp[1]), 5, color)
        
        # Draw current window position
        current_bounds = (
            max(0, current_cy - half_h),
            min(H, current_cy + half_h),
            max(0, current_cx - half_w),
            min(W, current_cx + half_w)
        )
        seg_color = SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)]
        scene_overlay = draw_rectangle(scene_overlay, current_bounds, seg_color, thickness=2)
        
        # Resize scene for side-by-side view
        target_h = win_h
        scale = target_h / H
        target_w = int(W * scale)
        
        if CV2_AVAILABLE:
            scene_resized = cv2.resize(scene_overlay, (target_w, target_h))
        else:
            h_idx = np.linspace(0, H - 1, target_h).astype(int)
            w_idx = np.linspace(0, W - 1, target_w).astype(int)
            scene_resized = scene_overlay[np.ix_(h_idx, w_idx)]
        
        # Create separator
        separator = np.ones((target_h, 4, 3), dtype=np.uint8) * 255
        
        # Combine: [scene] | [separator] | [window]
        combined = np.concatenate([scene_resized, separator, frame], axis=1)
        
        # Add text annotations
        current_angle = segment_angles[seg_idx] if seg_idx < len(segment_angles) else segment_angles[-1]
        progress = (frame_idx + 1) / len(sliding_frames) * 100
        
        text1 = f"Frame {frame_idx+1}/{len(sliding_frames)} | Segment {seg_idx+1}/{num_segments} | Angle: {current_angle:.0f} deg"
        combined = add_text_to_image(combined, text1, (5, 5))
        
        text2 = f"Progress: {progress:.0f}% | Overlap: {overlap_ratio:.1%}"
        combined = add_text_to_image(combined, text2, (5, target_h - 18))
        
        frames_for_gif.append(combined)
    
    print(f"\nSaving GIF to: {output_path}")
    
    try:
        imageio.mimsave(output_path, frames_for_gif, fps=fps, loop=0)
        print(f"Success! GIF saved: {output_path}")
        print(f"  - Frames: {len(frames_for_gif)}")
        print(f"  - Size: {frames_for_gif[0].shape[1]}x{frames_for_gif[0].shape[0]}")
        return True
    except Exception as e:
        print(f"Error saving GIF: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Visualize Multi-Direction Benchmark as GIF")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input HDF5 file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output GIF file")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Sample index")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()
    
    success = create_multi_direction_gif(
        input_path=args.input,
        output_path=args.output,
        sample_idx=args.sample,
        fps=args.fps
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
