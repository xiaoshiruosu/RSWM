#!/usr/bin/env python3
"""
Multi-Direction Sliding Window MP4 Visualizer (V4)

Visualizes the multi-directional sliding window path as MP4 video instead of GIF.
Shows the window following a path with multiple direction changes.

Features:
- Full scene view with trajectory overlay
- Current window position with direction indicator
- Segment-by-segment color coding
- MP4 output format for better quality and smaller file size
"""

import os
import sys
import argparse
import numpy as np
import h5py

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Error: OpenCV required for MP4 output. Install with: pip install opencv-python")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


DEFAULT_INPUT = "multi_direction_benchmark_v3/benchmark.h5"
DEFAULT_OUTPUT = "multi_direction_demo.mp4"

# Segment colors (BGR for OpenCV)
SEGMENT_COLORS_BGR = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

# RGB version for PIL
SEGMENT_COLORS_RGB = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 (0-255)"""
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        return (arr * 255).astype(np.uint8)
    return arr.clip(0, 255).astype(np.uint8)


def draw_rectangle_cv2(image: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw rectangle using OpenCV"""
    img = image.copy()
    cv2.rectangle(img, pt1, pt2, color, thickness)
    return img


def draw_line_cv2(image: np.ndarray, pt1: tuple, pt2: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw line using OpenCV"""
    img = image.copy()
    cv2.line(img, pt1, pt2, color, thickness)
    return img


def draw_circle_cv2(image: np.ndarray, center: tuple, radius: int, color: tuple, thickness: int = -1) -> np.ndarray:
    """Draw circle using OpenCV"""
    img = image.copy()
    cv2.circle(img, center, radius, color, thickness)
    return img


def put_text_cv2(image: np.ndarray, text: str, position: tuple, color: tuple, scale: float = 0.5, thickness: int = 1) -> np.ndarray:
    """Put text on image using OpenCV"""
    img = image.copy()
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img


class MultiDirectionMP4Visualizer:
    """Visualize multi-direction benchmark as MP4 video"""
    
    def __init__(self, h5_path: str, output_path: str, fps: int = 10, codec: str = "mp4v"):
        self.h5_path = h5_path
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.h5_file = None
    
    def __enter__(self):
        self.h5_file = h5py.File(self.h5_path, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.h5_file:
            self.h5_file.close()
    
    def get_sample_count(self) -> int:
        """Get total number of samples"""
        return self.h5_file['metadata'].attrs['total_instances']
    
    def load_sample(self, sample_idx: int) -> dict:
        """Load a sample from HDF5"""
        sample_key = f'data/sample_{sample_idx:05d}'
        if sample_key not in self.h5_file:
            raise ValueError(f"Sample {sample_idx} not found")
        
        grp = self.h5_file[sample_key]
        
        return {
            'sliding_frames': grp['sliding_frames'][:],
            'center_frame': grp['center_frame'][:],
            'target_frame': grp['target_frame'][:],
            'full_scene': grp['full_scene'][:],
            'direction_embedding': grp['direction_embedding'][:],
            'trajectory': grp['trajectory'][:],
            'segment_angles': grp['segment_angles'][:],
            'waypoints': grp['waypoints'][:],
            'num_segments': grp.attrs['num_segments'],
            'scene_category': grp.attrs['scene_category'],
            'sample_id': grp.attrs['sample_id'],
        }
    
    def create_visualization_frame(
        self,
        full_scene: np.ndarray,
        current_window: np.ndarray,
        trajectory: np.ndarray,
        current_frame_idx: int,
        waypoints: np.ndarray,
        segment_angles: np.ndarray,
        window_size: tuple,
        steps_per_segment: int
    ) -> np.ndarray:
        """Create a single visualization frame"""
        # Normalize images
        scene = normalize_to_uint8(full_scene.copy())
        window = normalize_to_uint8(current_window.copy())
        
        # Convert to BGR for OpenCV
        if scene.shape[2] == 3:
            scene_bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
        else:
            scene_bgr = scene
        
        if window.shape[2] == 3:
            window_bgr = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
        else:
            window_bgr = window
        
        # Draw trajectory path on scene
        num_segments = len(segment_angles)
        for seg_idx in range(num_segments):
            color = SEGMENT_COLORS_BGR[seg_idx % len(SEGMENT_COLORS_BGR)]
            start_wp = waypoints[seg_idx]
            end_wp = waypoints[seg_idx + 1]
            
            # Draw segment line
            pt1 = (int(start_wp[1]), int(start_wp[0]))
            pt2 = (int(end_wp[1]), int(end_wp[0]))
            scene_bgr = draw_line_cv2(scene_bgr, pt1, pt2, color, 2)
            
            # Draw waypoint circles
            scene_bgr = draw_circle_cv2(scene_bgr, pt1, 5, color, -1)
            scene_bgr = draw_circle_cv2(scene_bgr, pt2, 5, color, -1)
        
        # Draw current window position
        if current_frame_idx < len(trajectory):
            cy, cx = trajectory[current_frame_idx]
            win_h, win_w = window_size
            
            # Calculate current segment
            current_segment = min(current_frame_idx // steps_per_segment, num_segments - 1)
            current_color = SEGMENT_COLORS_BGR[current_segment % len(SEGMENT_COLORS_BGR)]
            
            # Draw current window rectangle
            x1 = int(cx - win_w // 2)
            y1 = int(cy - win_h // 2)
            x2 = int(cx + win_w // 2)
            y2 = int(cy + win_h // 2)
            scene_bgr = draw_rectangle_cv2(scene_bgr, (x1, y1), (x2, y2), current_color, 3)
            
            # Draw current position marker
            scene_bgr = draw_circle_cv2(scene_bgr, (int(cx), int(cy)), 8, (255, 255, 255), -1)
            scene_bgr = draw_circle_cv2(scene_bgr, (int(cx), int(cy)), 6, current_color, -1)
        
        # Add text info
        scene_bgr = put_text_cv2(scene_bgr, f"Frame: {current_frame_idx + 1}/{len(trajectory)}", (10, 25), (255, 255, 255), 0.6, 2)
        if current_frame_idx < len(trajectory):
            current_segment = min(current_frame_idx // steps_per_segment, num_segments - 1)
            angle = segment_angles[current_segment]
            scene_bgr = put_text_cv2(scene_bgr, f"Segment: {current_segment + 1}/{num_segments}", (10, 50), (255, 255, 255), 0.6, 2)
            scene_bgr = put_text_cv2(scene_bgr, f"Angle: {angle:.1f} deg", (10, 75), (255, 255, 255), 0.6, 2)
        
        # Resize window to match scene height
        scene_h, scene_w = scene_bgr.shape[:2]
        win_h, win_w = window_bgr.shape[:2]
        
        scale = scene_h / win_h
        new_win_w = int(win_w * scale)
        window_resized = cv2.resize(window_bgr, (new_win_w, scene_h), interpolation=cv2.INTER_LINEAR)
        
        # Add border to window
        window_resized = cv2.copyMakeBorder(window_resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Add "Current Window" label
        window_resized = put_text_cv2(window_resized, "Current Window", (10, 25), (255, 255, 255), 0.6, 2)
        
        # Combine scene and window side by side
        combined = np.hstack([scene_bgr, window_resized[:scene_h, :]])
        
        return combined
    
    def visualize_sample(self, sample_idx: int, output_path: str = None) -> str:
        """Visualize a single sample as MP4"""
        if output_path is None:
            output_path = self.output_path
        
        sample = self.load_sample(sample_idx)
        
        sliding_frames = sample['sliding_frames']
        full_scene = sample['full_scene']
        trajectory = sample['trajectory']
        waypoints = sample['waypoints']
        segment_angles = sample['segment_angles']
        
        # Get window size from first frame
        window_size = (sliding_frames.shape[1], sliding_frames.shape[2])
        
        # Calculate steps per segment
        num_segments = sample['num_segments']
        total_frames = len(sliding_frames)
        steps_per_segment = total_frames // num_segments if num_segments > 0 else total_frames
        
        # Create first frame to get dimensions
        first_frame = self.create_visualization_frame(
            full_scene, sliding_frames[0], trajectory, 0,
            waypoints, segment_angles, window_size, steps_per_segment
        )
        
        h, w = first_frame.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Write frames
        for frame_idx in range(total_frames):
            vis_frame = self.create_visualization_frame(
                full_scene, sliding_frames[frame_idx], trajectory, frame_idx,
                waypoints, segment_angles, window_size, steps_per_segment
            )
            writer.write(vis_frame)
        
        writer.release()
        print(f"Saved visualization to {output_path}")
        
        return output_path
    
    def visualize_window_only(self, sample_idx: int, output_path: str = None) -> str:
        """Visualize only the sliding window (no scene overlay) as MP4"""
        if output_path is None:
            base, ext = os.path.splitext(self.output_path)
            output_path = f"{base}_window_only{ext}"
        
        sample = self.load_sample(sample_idx)
        sliding_frames = sample['sliding_frames']
        
        # Get dimensions
        n, h, w, c = sliding_frames.shape
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Write frames
        for frame_idx in range(n):
            frame = normalize_to_uint8(sliding_frames[frame_idx])
            if c == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)
        
        writer.release()
        print(f"Saved window-only video to {output_path}")
        
        return output_path
    
    def batch_visualize(self, sample_indices: list, output_dir: str, window_only: bool = False) -> list:
        """Batch visualize multiple samples"""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for idx in sample_indices:
            if window_only:
                output_path = os.path.join(output_dir, f"sample_{idx:05d}_window.mp4")
                self.visualize_window_only(idx, output_path)
            else:
                output_path = os.path.join(output_dir, f"sample_{idx:05d}.mp4")
                self.visualize_sample(idx, output_path)
            output_paths.append(output_path)
        
        return output_paths


def main():
    if not CV2_AVAILABLE:
        print("Error: OpenCV is required for MP4 output.")
        print("Install with: pip install opencv-python")
        return 1
    
    parser = argparse.ArgumentParser(description="Visualize Multi-Direction Benchmark as MP4")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input HDF5 file")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output MP4 file")
    parser.add_argument("--sample", "-s", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--window-only", action="store_true", help="Only show sliding window (no scene)")
    parser.add_argument("--batch", type=int, nargs='+', help="Batch visualize multiple samples")
    parser.add_argument("--batch-dir", default="visualizations", help="Output directory for batch")
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.input):
        # Try alternative paths
        alt_paths = [
            "multi_direction_benchmark_v3/benchmark.h5",
            "multi_direction_benchmark.h5",
            "multi_direction_benchmark_v2.h5",
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                args.input = alt
                print(f"Using input file: {args.input}")
                break
        else:
            print(f"Error: Input file not found: {args.input}")
            return 1
    
    with MultiDirectionMP4Visualizer(args.input, args.output, fps=args.fps) as viz:
        total_samples = viz.get_sample_count()
        print(f"Total samples in benchmark: {total_samples}")
        
        if args.batch:
            # Batch mode
            viz.batch_visualize(args.batch, args.batch_dir, window_only=args.window_only)
        else:
            # Single sample mode
            if args.sample >= total_samples:
                print(f"Error: Sample index {args.sample} out of range (0-{total_samples-1})")
                return 1
            
            if args.window_only:
                viz.visualize_window_only(args.sample, args.output)
            else:
                viz.visualize_sample(args.sample, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
