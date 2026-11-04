"""
Validation Module for RSVideoWise Benchmark

Validates:
1. Overlap ratio between center and target sequences (<10%)
2. Direction accuracy and consistency
3. Sequence quality and format
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    overlap_valid: bool
    direction_valid: bool
    format_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, errors={len(self.errors)}, warnings={len(self.warnings)})"


class OverlapValidator:
    """
    Validate overlap between center and target sequences.
    
    Ensures overlap ratio is below threshold (default <10%).
    """
    
    def __init__(
        self,
        max_overlap_ratio: float = 0.10,
        strict_mode: bool = True
    ):
        """
        Initialize overlap validator.
        
        Args:
            max_overlap_ratio: Maximum allowed overlap ratio
            strict_mode: If True, fail on any violation
        """
        self.max_overlap_ratio = max_overlap_ratio
        self.strict_mode = strict_mode
    
    def validate(
        self,
        center_bounds: Tuple[int, int, int, int],
        target_bounds: Tuple[int, int, int, int]
    ) -> Tuple[bool, float, str]:
        """
        Validate overlap between center and target regions.
        
        Args:
            center_bounds: (h_start, h_end, w_start, w_end) for center
            target_bounds: (h_start, h_end, w_start, w_end) for target
            
        Returns:
            Tuple of (is_valid, overlap_ratio, message)
        """
        overlap_ratio = self._compute_overlap(center_bounds, target_bounds)
        
        is_valid = overlap_ratio < self.max_overlap_ratio
        
        if is_valid:
            message = f"Overlap {overlap_ratio:.2%} is within threshold {self.max_overlap_ratio:.0%}"
        else:
            message = f"Overlap {overlap_ratio:.2%} exceeds threshold {self.max_overlap_ratio:.0%}"
        
        return is_valid, overlap_ratio, message
    
    def _compute_overlap(
        self,
        bounds1: Tuple[int, int, int, int],
        bounds2: Tuple[int, int, int, int]
    ) -> float:
        """Compute intersection over area ratio"""
        h1_s, h1_e, w1_s, w1_e = bounds1
        h2_s, h2_e, w2_s, w2_e = bounds2
        
        # Compute intersection
        h_inter_start = max(h1_s, h2_s)
        h_inter_end = min(h1_e, h2_e)
        w_inter_start = max(w1_s, w2_s)
        w_inter_end = min(w1_e, w2_e)
        
        if h_inter_start >= h_inter_end or w_inter_start >= w_inter_end:
            return 0.0
        
        intersection_area = (h_inter_end - h_inter_start) * (w_inter_end - w_inter_start)
        center_area = (h1_e - h1_s) * (w1_e - w1_s)
        
        return intersection_area / center_area if center_area > 0 else 0.0
    
    def validate_batch(
        self,
        samples: List[Dict]
    ) -> Tuple[int, int, List[float]]:
        """
        Validate batch of samples.
        
        Returns:
            Tuple of (valid_count, invalid_count, overlap_ratios)
        """
        valid_count = 0
        invalid_count = 0
        overlap_ratios = []
        
        for sample in samples:
            center_bounds = sample.get("center_bounds")
            target_bounds = sample.get("target_bounds")
            
            if center_bounds is None or target_bounds is None:
                invalid_count += 1
                continue
            
            is_valid, ratio, _ = self.validate(center_bounds, target_bounds)
            overlap_ratios.append(ratio)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        return valid_count, invalid_count, overlap_ratios


class DirectionValidator:
    """
    Validate direction accuracy and consistency.
    
    Ensures:
    - Direction embeddings are properly normalized
    - Direction vectors point to actual target location
    - Angles are within valid range
    """
    
    def __init__(
        self,
        angle_tolerance: float = 15.0  # degrees
    ):
        """
        Initialize direction validator.

        Args:
            angle_tolerance: Maximum angle deviation allowed (degrees)
        """
        self.angle_tolerance = angle_tolerance

    def validate(
        self,
        claimed_angle: float,
        center_bounds: Tuple[int, int, int, int],
        target_bounds: Tuple[int, int, int, int]
    ) -> Tuple[bool, float, str]:
        """
        Validate that claimed direction matches actual displacement.

        Args:
            claimed_angle: Claimed direction angle in degrees
            center_bounds: Center region bounds
            target_bounds: Target region bounds

        Returns:
            Tuple of (is_valid, actual_angle, message)
        """
        # Compute actual displacement
        actual_angle = self._compute_actual_angle(center_bounds, target_bounds)

        # Compute angle difference (handle wrap-around)
        angle_diff = self._angle_difference(claimed_angle, actual_angle)

        is_valid = angle_diff <= self.angle_tolerance

        if is_valid:
            message = f"Direction accurate within {angle_diff:.1f}° (tolerance: {self.angle_tolerance}°)"
        else:
            message = f"Direction error {angle_diff:.1f}° exceeds tolerance {self.angle_tolerance}°"

        return is_valid, actual_angle, message

    def _compute_actual_angle(
        self,
        center_bounds: Tuple[int, int, int, int],
        target_bounds: Tuple[int, int, int, int]
    ) -> float:
        """Compute actual direction angle from center to target"""
        # Get centers of both regions
        c_h = (center_bounds[0] + center_bounds[1]) / 2
        c_w = (center_bounds[2] + center_bounds[3]) / 2
        t_h = (target_bounds[0] + target_bounds[1]) / 2
        t_w = (target_bounds[2] + target_bounds[3]) / 2

        # Compute displacement
        dw = t_w - c_w  # Positive = right
        dh = c_h - t_h  # Positive = up (invert because row increases downward)

        # Convert to angle (0° = right, 90° = up)
        angle_rad = np.arctan2(dh, dw)
        angle_deg = np.rad2deg(angle_rad)

        # Normalize to [0, 360)
        return angle_deg % 360

    def _angle_difference(
        self,
        angle1: float,
        angle2: float
    ) -> float:
        """Compute minimum angle difference (handles wrap-around)"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    def validate_embedding(
        self,
        embedding: np.ndarray,
        expected_dim: int = 64
    ) -> Tuple[bool, str]:
        """
        Validate embedding format and properties.

        Args:
            embedding: Direction embedding array
            expected_dim: Expected embedding dimension

        Returns:
            Tuple of (is_valid, message)
        """
        # Check dimension
        if embedding.shape[-1] != expected_dim:
            return False, f"Expected dim {expected_dim}, got {embedding.shape[-1]}"

        # Check for NaN/Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False, "Embedding contains NaN or Inf values"

        # Check normalization (should have reasonable magnitude)
        norm = np.linalg.norm(embedding)
        if norm < 0.1 or norm > 100:
            return False, f"Unusual embedding norm: {norm:.2f}"

        return True, "Embedding format valid"


class SequenceValidator:
    """Validate sequence format and quality"""

    def __init__(
        self,
        min_frames: int = 8,
        max_frames: int = 16,
        expected_channels: int = 3
    ):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.expected_channels = expected_channels

    def validate(
        self,
        sequence: np.ndarray,
        sequence_name: str = "sequence"
    ) -> Tuple[bool, List[str]]:
        """
        Validate sequence format.

        Args:
            sequence: Sequence array (T, H, W, C)
            sequence_name: Name for error messages

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check dimensions
        if len(sequence.shape) != 4:
            issues.append(f"{sequence_name}: Expected 4D, got {len(sequence.shape)}D")
            return False, issues

        T, H, W, C = sequence.shape

        # Check temporal length
        if T < self.min_frames:
            issues.append(f"{sequence_name}: Too few frames ({T} < {self.min_frames})")
        if T > self.max_frames:
            issues.append(f"{sequence_name}: Too many frames ({T} > {self.max_frames})")

        # Check channels
        if C != self.expected_channels:
            issues.append(f"{sequence_name}: Expected {self.expected_channels} channels, got {C}")

        # Check for valid values
        if np.any(np.isnan(sequence)):
            issues.append(f"{sequence_name}: Contains NaN values")
        if np.any(np.isinf(sequence)):
            issues.append(f"{sequence_name}: Contains Inf values")

        return len(issues) == 0, issues


def validate_sample(
    sample: Dict,
    max_overlap: float = 0.10,
    angle_tolerance: float = 15.0
) -> ValidationResult:
    """
    Comprehensive validation of a single sample.

    Args:
        sample: Sample dictionary with sequences and metadata
        max_overlap: Maximum allowed overlap ratio
        angle_tolerance: Maximum angle deviation (degrees)

    Returns:
        ValidationResult with all validation outcomes
    """
    errors = []
    warnings = []
    metrics = {}

    # Validate overlap
    overlap_validator = OverlapValidator(max_overlap_ratio=max_overlap)
    overlap_valid, overlap_ratio, overlap_msg = overlap_validator.validate(
        sample.get("center_bounds", (0, 0, 0, 0)),
        sample.get("target_bounds", (0, 0, 0, 0))
    )
    metrics["overlap_ratio"] = overlap_ratio
    if not overlap_valid:
        errors.append(overlap_msg)

    # Validate direction
    direction_validator = DirectionValidator(angle_tolerance=angle_tolerance)
    direction_valid, actual_angle, direction_msg = direction_validator.validate(
        sample.get("direction_angle", 0),
        sample.get("center_bounds", (0, 0, 0, 0)),
        sample.get("target_bounds", (0, 0, 0, 0))
    )
    metrics["actual_angle"] = actual_angle
    metrics["claimed_angle"] = sample.get("direction_angle", 0)
    if not direction_valid:
        warnings.append(direction_msg)

    # Validate sequences
    seq_validator = SequenceValidator()
    format_valid = True

    if "center_sequence" in sample:
        valid, issues = seq_validator.validate(sample["center_sequence"], "center")
        if not valid:
            format_valid = False
            errors.extend(issues)

    if "target_sequence" in sample:
        valid, issues = seq_validator.validate(sample["target_sequence"], "target")
        if not valid:
            format_valid = False
            errors.extend(issues)

    is_valid = overlap_valid and format_valid and len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        overlap_valid=overlap_valid,
        direction_valid=direction_valid,
        format_valid=format_valid,
        errors=errors,
        warnings=warnings,
        metrics=metrics
    )

