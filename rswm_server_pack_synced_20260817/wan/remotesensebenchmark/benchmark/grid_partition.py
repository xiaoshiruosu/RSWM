"""
Spatiotemporal Grid Partitioning Module for RSVideoWise Benchmark

Divides input sequences into 3x3xT spatiotemporal grid with minimal overlap,
extracting center sequence and enabling directional target extraction.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GridCell:
    """Represents a single cell in the spatiotemporal grid"""
    row: int
    col: int
    temporal_start: int
    temporal_end: int
    spatial_slice: Tuple[slice, slice]
    data: Optional[np.ndarray] = None
    
    @property
    def position(self) -> Tuple[int, int]:
        return (self.row, self.col)
    
    @property
    def temporal_range(self) -> Tuple[int, int]:
        return (self.temporal_start, self.temporal_end)


@dataclass
class PartitionedGrid:
    """Container for a partitioned spatiotemporal grid"""
    cells: Dict[Tuple[int, int], GridCell]
    grid_size: Tuple[int, int]
    temporal_segments: int
    original_shape: Tuple[int, ...]
    cell_spatial_size: Tuple[int, int]
    cell_temporal_size: int
    
    def get_center_cell(self) -> GridCell:
        """Get the center cell (1, 1) in 3x3 grid"""
        center_row = self.grid_size[0] // 2
        center_col = self.grid_size[1] // 2
        return self.cells[(center_row, center_col)]
    
    def get_cell(self, row: int, col: int) -> Optional[GridCell]:
        """Get cell at specified position"""
        return self.cells.get((row, col))
    
    def get_neighbor_cells(self, row: int, col: int) -> List[GridCell]:
        """Get all valid neighboring cells (8-connected)"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (nr, nc) in self.cells:
                    neighbors.append(self.cells[(nr, nc)])
        return neighbors


class SpatioTemporalGridPartitioner:
    """
    Partition temporal sequences into 3x3xT spatiotemporal grids.
    
    The partitioner divides:
    - Spatial dimensions into 3x3 grid
    - Temporal dimension into T segments
    
    Center cell is used for V_c (center sequence)
    Directional cells are used for V_t (target sequence)
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (3, 3),
        min_overlap: float = 0.0,
        max_overlap: float = 0.05  # 5% spatial overlap between cells
    ):
        self.grid_size = grid_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        
    def partition(
        self,
        sequence: np.ndarray,
        num_temporal_segments: int = 1
    ) -> PartitionedGrid:
        """
        Partition a temporal sequence into spatiotemporal grid.
        
        Args:
            sequence: Input sequence of shape (T, H, W, C) or (T, C, H, W)
            num_temporal_segments: Number of temporal divisions
            
        Returns:
            PartitionedGrid containing all grid cells
        """
        # Determine data format and extract dimensions
        if len(sequence.shape) == 4:
            T, dim1, dim2, dim3 = sequence.shape
            # Heuristic: if dim3 is small (<=4), assume (T, H, W, C)
            if dim3 <= 4:
                H, W, C = dim1, dim2, dim3
                data_format = "THWC"
            else:
                C, H, W = dim1, dim2, dim3
                data_format = "TCHW"
        else:
            raise ValueError(f"Expected 4D sequence, got shape {sequence.shape}")
        
        logger.debug(f"Partitioning sequence: T={T}, H={H}, W={W}, format={data_format}")
        
        # Calculate cell sizes with overlap
        overlap_h = int(H * self.max_overlap / (self.grid_size[0] - 1)) if self.grid_size[0] > 1 else 0
        overlap_w = int(W * self.max_overlap / (self.grid_size[1] - 1)) if self.grid_size[1] > 1 else 0
        
        # Cell size without overlap
        cell_h = (H + overlap_h * (self.grid_size[0] - 1)) // self.grid_size[0]
        cell_w = (W + overlap_w * (self.grid_size[1] - 1)) // self.grid_size[1]
        
        # Temporal segment size
        cell_t = T // num_temporal_segments
        
        cells = {}
        
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Calculate spatial boundaries
                h_start = row * (cell_h - overlap_h)
                h_end = min(h_start + cell_h, H)
                w_start = col * (cell_w - overlap_w)
                w_end = min(w_start + cell_w, W)
                
                spatial_slice = (slice(h_start, h_end), slice(w_start, w_end))
                
                # Extract cell data for first temporal segment
                if data_format == "THWC":
                    cell_data = sequence[:cell_t, h_start:h_end, w_start:w_end, :]
                else:
                    cell_data = sequence[:cell_t, :, h_start:h_end, w_start:w_end]
                
                cells[(row, col)] = GridCell(
                    row=row,
                    col=col,
                    temporal_start=0,
                    temporal_end=cell_t,
                    spatial_slice=spatial_slice,
                    data=cell_data
                )
        
        return PartitionedGrid(
            cells=cells,
            grid_size=self.grid_size,
            temporal_segments=num_temporal_segments,
            original_shape=sequence.shape,
            cell_spatial_size=(cell_h, cell_w),
            cell_temporal_size=cell_t
        )

    def extract_center_sequence(
        self,
        sequence: np.ndarray,
        num_frames: int = 16
    ) -> Tuple[np.ndarray, Dict]:
        """
        Extract center sequence V_c from the spatiotemporal grid.

        Args:
            sequence: Input sequence of shape (T, H, W, C)
            num_frames: Number of frames to extract (8-16)

        Returns:
            Tuple of (center_sequence, metadata)
        """
        T, H, W, C = sequence.shape

        # Center spatial region (middle third of each dimension)
        h_start = H // 3
        h_end = 2 * H // 3
        w_start = W // 3
        w_end = 2 * W // 3

        # Select temporal window
        num_frames = min(num_frames, T)
        t_start = (T - num_frames) // 2
        t_end = t_start + num_frames

        center_seq = sequence[t_start:t_end, h_start:h_end, w_start:w_end, :]

        metadata = {
            "spatial_bounds": (h_start, h_end, w_start, w_end),
            "temporal_bounds": (t_start, t_end),
            "center_position": (H // 2, W // 2),
            "num_frames": num_frames
        }

        return center_seq, metadata

    def get_direction_offset(
        self,
        angle_degrees: float,
        magnitude: float = 1.0
    ) -> Tuple[float, float]:
        """
        Convert angle to grid offset direction.

        Args:
            angle_degrees: Angle in degrees (0-360, 0=right, 90=up)
            magnitude: Step size multiplier

        Returns:
            (row_offset, col_offset) normalized direction
        """
        angle_rad = np.deg2rad(angle_degrees)

        # Convert to grid coordinates (row increases downward)
        col_offset = np.cos(angle_rad) * magnitude
        row_offset = -np.sin(angle_rad) * magnitude  # Negative because row increases downward

        return (row_offset, col_offset)

    def get_target_cell_position(
        self,
        center_pos: Tuple[int, int],
        angle_degrees: float
    ) -> Tuple[int, int]:
        """
        Get target grid cell position based on direction angle.

        Maps continuous angle to one of 8 neighboring cells.
        """
        # Discretize angle to 8 directions (N, NE, E, SE, S, SW, W, NW)
        angle = angle_degrees % 360

        # Direction mapping (angle ranges to grid offsets)
        direction_map = [
            (337.5, 22.5, (0, 1)),    # E: 0°
            (22.5, 67.5, (-1, 1)),    # NE: 45°
            (67.5, 112.5, (-1, 0)),   # N: 90°
            (112.5, 157.5, (-1, -1)), # NW: 135°
            (157.5, 202.5, (0, -1)),  # W: 180°
            (202.5, 247.5, (1, -1)),  # SW: 225°
            (247.5, 292.5, (1, 0)),   # S: 270°
            (292.5, 337.5, (1, 1)),   # SE: 315°
        ]

        for start, end, offset in direction_map:
            if start <= angle < end or (start > end and (angle >= start or angle < end)):
                target_row = center_pos[0] + offset[0]
                target_col = center_pos[1] + offset[1]
                return (target_row, target_col)

        # Default to East
        return (center_pos[0], center_pos[1] + 1)

    def compute_overlap_ratio(
        self,
        region1: Tuple[int, int, int, int],
        region2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute overlap ratio between two spatial regions.

        Args:
            region1, region2: (h_start, h_end, w_start, w_end)

        Returns:
            Overlap ratio (intersection / union)
        """
        h1_s, h1_e, w1_s, w1_e = region1
        h2_s, h2_e, w2_s, w2_e = region2

        # Compute intersection
        h_inter_s = max(h1_s, h2_s)
        h_inter_e = min(h1_e, h2_e)
        w_inter_s = max(w1_s, w2_s)
        w_inter_e = min(w1_e, w2_e)

        if h_inter_s >= h_inter_e or w_inter_s >= w_inter_e:
            return 0.0

        intersection = (h_inter_e - h_inter_s) * (w_inter_e - w_inter_s)

        area1 = (h1_e - h1_s) * (w1_e - w1_s)
        area2 = (h2_e - h2_s) * (w2_e - w2_s)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

