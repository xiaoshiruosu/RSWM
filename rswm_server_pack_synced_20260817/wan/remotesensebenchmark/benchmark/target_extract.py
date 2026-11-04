"""
Target Sequence Extraction Module for RSVideoWise Benchmark

Extracts target sequences V_t along specified directions with minimal overlap (<10%)
to the center sequence V_c. Supports interpolation for arbitrary angles.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import cv2 for interpolation
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available, using scipy for interpolation")

try:
    from scipy.ndimage import map_coordinates, shift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TargetExtractionResult:
    """Container for target sequence extraction result"""
    target_sequence: np.ndarray
    center_sequence: np.ndarray
    direction_angle: float
    overlap_ratio: float
    extraction_method: str
    metadata: Dict
    
    @property
    def is_valid(self) -> bool:
        """Check if extraction meets overlap constraint"""
        return self.overlap_ratio < 0.10
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "target_sequence": self.target_sequence,
            "center_sequence": self.center_sequence,
            "direction_angle": self.direction_angle,
            "overlap_ratio": self.overlap_ratio,
            "extraction_method": self.extraction_method,
            "metadata": self.metadata
        }


class TargetSequenceExtractor:
    """
    Extract target sequences along arbitrary directions with minimal overlap.
    
    Key features:
    - Supports continuous angles (0-360 degrees)
    - Bilinear interpolation for sub-pixel accuracy
    - Overlap ratio validation (<10%)
    - Multiple extraction strategies (grid-based, continuous)
    """
    
    def __init__(
        self,
        max_overlap_ratio: float = 0.10,
        min_frames: int = 8,
        max_frames: int = 16,
        interpolation_method: str = "bilinear"
    ):
        """
        Initialize target sequence extractor.
        
        Args:
            max_overlap_ratio: Maximum allowed overlap with center sequence
            min_frames: Minimum frames in extracted sequence
            max_frames: Maximum frames in extracted sequence
            interpolation_method: Interpolation method ('nearest', 'bilinear')
        """
        self.max_overlap_ratio = max_overlap_ratio
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interpolation_method = interpolation_method
    
    def extract(
        self,
        full_sequence: np.ndarray,
        center_bounds: Tuple[int, int, int, int],
        direction_angle: float,
        num_frames: int = 16
    ) -> TargetExtractionResult:
        """
        Extract target sequence along specified direction.
        
        Args:
            full_sequence: Full input sequence (T, H, W, C)
            center_bounds: Center region bounds (h_start, h_end, w_start, w_end)
            direction_angle: Extraction direction in degrees
            num_frames: Number of frames to extract
            
        Returns:
            TargetExtractionResult with extracted sequences and metadata
        """
        T, H, W, C = full_sequence.shape
        h_start, h_end, w_start, w_end = center_bounds
        
        # Calculate center region dimensions
        center_h = h_end - h_start
        center_w = w_end - w_start
        
        # Extract center sequence
        num_frames = min(num_frames, T)
        t_start = (T - num_frames) // 2
        t_end = t_start + num_frames
        center_sequence = full_sequence[t_start:t_end, h_start:h_end, w_start:w_end, :]
        
        # Calculate target region based on direction
        target_bounds = self._calculate_target_bounds(
            center_bounds, direction_angle, (H, W), (center_h, center_w)
        )
        
        # Extract target sequence
        th_start, th_end, tw_start, tw_end = target_bounds
        target_sequence = full_sequence[t_start:t_end, th_start:th_end, tw_start:tw_end, :]
        
        # Resize target to match center dimensions if needed
        if target_sequence.shape[1:3] != center_sequence.shape[1:3]:
            target_sequence = self._resize_sequence(
                target_sequence, (center_h, center_w)
            )
        
        # Calculate overlap ratio
        overlap_ratio = self._calculate_overlap(center_bounds, target_bounds)
        
        # Validate and potentially adjust if overlap too high
        if overlap_ratio >= self.max_overlap_ratio:
            target_sequence, target_bounds, overlap_ratio = self._adjust_for_overlap(
                full_sequence, center_bounds, direction_angle,
                num_frames, t_start, t_end, (H, W)
            )
        
        metadata = {
            "center_bounds": center_bounds,
            "target_bounds": target_bounds,
            "temporal_range": (t_start, t_end),
            "original_shape": full_sequence.shape
        }
        
        return TargetExtractionResult(
            target_sequence=target_sequence,
            center_sequence=center_sequence,
            direction_angle=direction_angle,
            overlap_ratio=overlap_ratio,
            extraction_method="grid_shift",
            metadata=metadata
        )

    def _calculate_target_bounds(
        self,
        center_bounds: Tuple[int, int, int, int],
        angle_degrees: float,
        image_size: Tuple[int, int],
        region_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Calculate target region bounds based on direction angle.

        Ensures minimal overlap with center region.
        """
        h_start, h_end, w_start, w_end = center_bounds
        H, W = image_size
        region_h, region_w = region_size

        # Convert angle to direction
        angle_rad = np.deg2rad(angle_degrees)
        dx = np.cos(angle_rad)  # Positive = right
        dy = -np.sin(angle_rad)  # Negative because y increases downward

        # Calculate shift to minimize overlap
        # Shift by region size in the direction
        shift_h = int(dy * region_h * 0.95)  # 95% shift for <10% overlap
        shift_w = int(dx * region_w * 0.95)

        # Calculate new bounds
        new_h_start = h_start + shift_h
        new_h_end = h_end + shift_h
        new_w_start = w_start + shift_w
        new_w_end = w_end + shift_w

        # Clamp to image boundaries
        if new_h_start < 0:
            new_h_end -= new_h_start
            new_h_start = 0
        if new_h_end > H:
            new_h_start -= (new_h_end - H)
            new_h_end = H
        if new_w_start < 0:
            new_w_end -= new_w_start
            new_w_start = 0
        if new_w_end > W:
            new_w_start -= (new_w_end - W)
            new_w_end = W

        # Ensure positive bounds
        new_h_start = max(0, new_h_start)
        new_w_start = max(0, new_w_start)
        new_h_end = min(H, max(new_h_start + 1, new_h_end))
        new_w_end = min(W, max(new_w_start + 1, new_w_end))

        return (new_h_start, new_h_end, new_w_start, new_w_end)

    def _calculate_overlap(
        self,
        bounds1: Tuple[int, int, int, int],
        bounds2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two regions"""
        h1_s, h1_e, w1_s, w1_e = bounds1
        h2_s, h2_e, w2_s, w2_e = bounds2

        # Intersection
        h_inter_s = max(h1_s, h2_s)
        h_inter_e = min(h1_e, h2_e)
        w_inter_s = max(w1_s, w2_s)
        w_inter_e = min(w1_e, w2_e)

        if h_inter_s >= h_inter_e or w_inter_s >= w_inter_e:
            return 0.0

        intersection = (h_inter_e - h_inter_s) * (w_inter_e - w_inter_s)
        area1 = (h1_e - h1_s) * (w1_e - w1_s)

        return intersection / area1 if area1 > 0 else 0.0

    def _resize_sequence(
        self,
        sequence: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize sequence to target spatial dimensions"""
        T, H, W, C = sequence.shape
        target_h, target_w = target_size

        if H == target_h and W == target_w:
            return sequence

        resized = np.zeros((T, target_h, target_w, C), dtype=sequence.dtype)

        for t in range(T):
            frame = sequence[t]
            if CV2_AVAILABLE:
                resized[t] = cv2.resize(
                    frame, (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
            elif SCIPY_AVAILABLE:
                # Use scipy zoom
                from scipy.ndimage import zoom
                factors = (target_h / H, target_w / W, 1)
                resized[t] = zoom(frame, factors, order=1)
            else:
                # Simple nearest neighbor
                h_idx = np.linspace(0, H - 1, target_h).astype(int)
                w_idx = np.linspace(0, W - 1, target_w).astype(int)
                resized[t] = frame[np.ix_(h_idx, w_idx)]

        return resized

    def _adjust_for_overlap(
        self,
        full_sequence: np.ndarray,
        center_bounds: Tuple[int, int, int, int],
        direction_angle: float,
        num_frames: int,
        t_start: int,
        t_end: int,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """Adjust extraction to reduce overlap below threshold"""
        H, W = image_size
        h_start, h_end, w_start, w_end = center_bounds
        region_h = h_end - h_start
        region_w = w_end - w_start

        # Try increasing shift until overlap is acceptable
        for shift_factor in [1.0, 1.1, 1.2, 1.3, 1.5]:
            angle_rad = np.deg2rad(direction_angle)
            dx = np.cos(angle_rad)
            dy = -np.sin(angle_rad)

            shift_h = int(dy * region_h * shift_factor)
            shift_w = int(dx * region_w * shift_factor)

            new_h_start = max(0, min(H - region_h, h_start + shift_h))
            new_h_end = new_h_start + region_h
            new_w_start = max(0, min(W - region_w, w_start + shift_w))
            new_w_end = new_w_start + region_w

            target_bounds = (new_h_start, new_h_end, new_w_start, new_w_end)
            overlap = self._calculate_overlap(center_bounds, target_bounds)

            if overlap < self.max_overlap_ratio:
                target_seq = full_sequence[
                    t_start:t_end,
                    new_h_start:new_h_end,
                    new_w_start:new_w_end,
                    :
                ]
                return target_seq, target_bounds, overlap

        # Fallback: use opposite corner
        if direction_angle < 180:
            new_h_start, new_h_end = 0, region_h
        else:
            new_h_start, new_h_end = H - region_h, H

        if 90 < direction_angle < 270:
            new_w_start, new_w_end = 0, region_w
        else:
            new_w_start, new_w_end = W - region_w, W

        target_bounds = (new_h_start, new_h_end, new_w_start, new_w_end)
        overlap = self._calculate_overlap(center_bounds, target_bounds)
        target_seq = full_sequence[
            t_start:t_end, new_h_start:new_h_end, new_w_start:new_w_end, :
        ]

        return target_seq, target_bounds, overlap
