"""
Direction Embedding Module for RSVideoWise Benchmark

Generates continuous sin/cos embeddings for arbitrary angles (0-360°),
supporting arbitrary direction continuous extrapolation.
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import torch, fall back to numpy implementation
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using NumPy for embeddings")


@dataclass
class DirectionEmbeddingResult:
    """Container for direction embedding output"""
    embedding: Union[np.ndarray, 'torch.Tensor']
    angle_degrees: float
    angle_radians: float
    magnitude: float
    direction_vector: Tuple[float, float]
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array"""
        if TORCH_AVAILABLE and isinstance(self.embedding, torch.Tensor):
            return self.embedding.cpu().numpy()
        return self.embedding
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        return {
            "embedding": self.to_numpy(),
            "angle_degrees": self.angle_degrees,
            "angle_radians": self.angle_radians,
            "magnitude": self.magnitude,
            "direction_vector": self.direction_vector
        }


class DirectionEmbedding:
    """
    Generate continuous direction embeddings using sinusoidal positional encoding.
    
    Supports:
    - Arbitrary angles (0-360 degrees)
    - Multiple frequency bands for fine-grained direction encoding
    - Optional magnitude encoding
    - Both numpy and PyTorch backends
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_freq_bands: int = 16,
        include_magnitude: bool = True,
        use_torch: bool = True
    ):
        """
        Initialize direction embedding generator.
        
        Args:
            embed_dim: Total embedding dimension
            num_freq_bands: Number of frequency bands for encoding
            include_magnitude: Whether to include magnitude information
            use_torch: Use PyTorch if available
        """
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        self.include_magnitude = include_magnitude
        self.use_torch = use_torch and TORCH_AVAILABLE
        
        # Calculate dimensions
        # Each frequency band produces 2 values (sin, cos)
        self.angle_dim = num_freq_bands * 2
        self.magnitude_dim = num_freq_bands if include_magnitude else 0
        
        # Direction vector (2D unit vector)
        self.direction_dim = 2
        
        # Total base dimension before projection
        self.base_dim = self.angle_dim + self.magnitude_dim + self.direction_dim
        
        # Frequency scales (log-spaced for multi-scale encoding)
        self.freq_scales = np.logspace(0, np.log10(1000), num_freq_bands)
        
        # Initialize projection layer if needed
        if self.use_torch and self.base_dim != embed_dim:
            self.projection = nn.Linear(self.base_dim, embed_dim)
        else:
            # Random projection matrix for numpy
            np.random.seed(42)
            self.projection_matrix = np.random.randn(self.base_dim, embed_dim)
            self.projection_matrix /= np.sqrt(self.base_dim)
    
    def encode_angle(
        self,
        angle_degrees: float
    ) -> np.ndarray:
        """
        Encode angle using sinusoidal positional encoding.
        
        Args:
            angle_degrees: Angle in degrees (0-360)
            
        Returns:
            Encoded angle as numpy array
        """
        angle_rad = np.deg2rad(angle_degrees)
        
        # Multi-frequency encoding
        encodings = []
        for freq in self.freq_scales:
            encodings.append(np.sin(angle_rad * freq))
            encodings.append(np.cos(angle_rad * freq))
        
        return np.array(encodings, dtype=np.float32)
    
    def encode_magnitude(
        self,
        magnitude: float,
        max_magnitude: float = 10.0
    ) -> np.ndarray:
        """
        Encode magnitude using sinusoidal encoding.
        
        Args:
            magnitude: Magnitude value
            max_magnitude: Maximum expected magnitude for normalization
            
        Returns:
            Encoded magnitude as numpy array
        """
        # Normalize magnitude to [0, 1]
        norm_mag = np.clip(magnitude / max_magnitude, 0, 1)
        
        # Multi-frequency encoding
        encodings = []
        for freq in self.freq_scales:
            encodings.append(np.sin(norm_mag * np.pi * freq))
        
        return np.array(encodings, dtype=np.float32)
    
    def get_direction_vector(
        self,
        angle_degrees: float
    ) -> Tuple[float, float]:
        """Convert angle to unit direction vector (dx, dy)"""
        angle_rad = np.deg2rad(angle_degrees)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        return (float(dx), float(dy))

    def generate(
        self,
        angle_degrees: float,
        magnitude: float = 1.0
    ) -> DirectionEmbeddingResult:
        """
        Generate complete direction embedding.

        Args:
            angle_degrees: Direction angle in degrees (0-360)
            magnitude: Optional magnitude/distance value

        Returns:
            DirectionEmbeddingResult containing embedding and metadata
        """
        # Normalize angle to [0, 360)
        angle_degrees = angle_degrees % 360
        angle_radians = np.deg2rad(angle_degrees)

        # Get components
        angle_encoding = self.encode_angle(angle_degrees)
        direction_vector = self.get_direction_vector(angle_degrees)

        # Build base embedding
        components = [angle_encoding, np.array(direction_vector, dtype=np.float32)]

        if self.include_magnitude:
            magnitude_encoding = self.encode_magnitude(magnitude)
            components.append(magnitude_encoding)

        base_embedding = np.concatenate(components)

        # Project to final dimension
        if self.use_torch:
            embedding = self._project_torch(base_embedding)
        else:
            embedding = self._project_numpy(base_embedding)

        return DirectionEmbeddingResult(
            embedding=embedding,
            angle_degrees=angle_degrees,
            angle_radians=angle_radians,
            magnitude=magnitude,
            direction_vector=direction_vector
        )

    def _project_numpy(self, base_embedding: np.ndarray) -> np.ndarray:
        """Project base embedding to target dimension using numpy"""
        if base_embedding.shape[0] == self.embed_dim:
            return base_embedding
        return np.dot(base_embedding, self.projection_matrix).astype(np.float32)

    def _project_torch(self, base_embedding: np.ndarray) -> 'torch.Tensor':
        """Project base embedding to target dimension using PyTorch"""
        if not TORCH_AVAILABLE:
            return self._project_numpy(base_embedding)

        tensor = torch.from_numpy(base_embedding).float()
        if hasattr(self, 'projection'):
            with torch.no_grad():
                return self.projection(tensor)
        return tensor

    def generate_batch(
        self,
        angles: np.ndarray,
        magnitudes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate embeddings for batch of angles.

        Args:
            angles: Array of angles in degrees
            magnitudes: Optional array of magnitudes

        Returns:
            Batch of embeddings, shape (N, embed_dim)
        """
        if magnitudes is None:
            magnitudes = np.ones_like(angles)

        embeddings = []
        for angle, mag in zip(angles, magnitudes):
            result = self.generate(float(angle), float(mag))
            embeddings.append(result.to_numpy())

        return np.stack(embeddings, axis=0)

    def sample_random_direction(
        self,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample random direction angles uniformly from [0, 360).

        Args:
            num_samples: Number of random angles to sample
            seed: Optional random seed

        Returns:
            Array of random angles in degrees
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(0, 360, size=num_samples).astype(np.float32)

    def interpolate_directions(
        self,
        angle_start: float,
        angle_end: float,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Interpolate between two directions.

        Uses shortest path interpolation on the circle.

        Args:
            angle_start: Starting angle in degrees
            angle_end: Ending angle in degrees
            num_steps: Number of interpolation steps

        Returns:
            Array of interpolated angles
        """
        # Normalize angles
        angle_start = angle_start % 360
        angle_end = angle_end % 360

        # Find shortest path
        diff = angle_end - angle_start
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360

        # Linear interpolation
        t = np.linspace(0, 1, num_steps)
        angles = (angle_start + t * diff) % 360

        return angles.astype(np.float32)

