"""
Safety constraints for the SafeDiffuser planning environment.

Classification: [ENGINEERING DECISION] — Higher-level constraint API wrapping CBFs.
The paper defines constraints via barrier functions; this module provides a user-facing
interface for defining obstacle constraints with world-coordinate positions.
"""

import torch
from typing import List, Tuple, Optional
from .barrier_functions import ControlBarrierFunction, EllipsoidBarrier, SuperellipseBarrier


class SafetyConstraint:
    """Base class for safety constraints in world coordinates."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def to_barrier(self, norm_mins: torch.Tensor, norm_maxs: torch.Tensor) -> ControlBarrierFunction:
        """Convert world-coordinate constraint to normalized-space barrier function.
        
        The upstream code normalizes coordinates to [-1, 1] using:
            x_norm = 2 * (x_world - 0.5 - norm_min) / (norm_max - norm_min) - 1
        
        Classification: [SPECIFIED] — Normalization from upstream code.
        """
        raise NotImplementedError


class ObstacleConstraint(SafetyConstraint):
    """Obstacle avoidance constraint in world coordinates.
    
    Classification: [SPECIFIED] — Matches upstream obstacle definitions.
    
    Args:
        name: Identifier for this obstacle
        center_world: (y, x) center in world coordinates
        radius_world: Radius in world coordinates
        shape: 'ellipse' (p=2) or 'superellipse' (p=4)
        scale: Additional scaling factors (ry_scale, rx_scale)
    """
    
    def __init__(self, name: str, center_world: Tuple[float, float],
                 radius_world: float = 1.0, shape: str = 'ellipse',
                 scale: Tuple[float, float] = (1.0, 1.0),
                 margin: float = 0.01, description: str = ""):
        super().__init__(name, description)
        self.center_world = center_world
        self.radius_world = radius_world
        self.shape = shape
        self.scale = scale
        self.margin = margin

    def to_barrier(self, norm_mins: torch.Tensor, norm_maxs: torch.Tensor) -> ControlBarrierFunction:
        """Convert to normalized-space barrier function.
        
        Classification: [SPECIFIED] — Normalization matches upstream code.
        Official repo: diffuser/models/diffusion.py, obstacle coordinate transformations.
        """
        cy_world, cx_world = self.center_world
        ry_scale, rx_scale = self.scale
        
        # Normalize center coordinates (upstream formula)
        cx_norm = 2 * (cx_world - 0.5 - norm_mins[1].item()) / (norm_maxs[1].item() - norm_mins[1].item()) - 1
        cy_norm = 2 * (cy_world - 0.5 - norm_mins[0].item()) / (norm_maxs[0].item() - norm_mins[0].item()) - 1
        
        # Normalize radii
        rx_norm = 2 * self.radius_world / (norm_maxs[1].item() - norm_mins[1].item()) * rx_scale
        ry_norm = 2 * self.radius_world / (norm_maxs[0].item() - norm_mins[0].item()) * ry_scale
        
        if self.shape == 'ellipse':
            return EllipsoidBarrier(
                center=(cy_norm, cx_norm),
                radii=(ry_norm, rx_norm),
                margin=self.margin
            )
        else:
            return SuperellipseBarrier(
                center=(cy_norm, cx_norm),
                radii=(ry_norm, rx_norm),
                power=4,
                margin=self.margin
            )

    def to_dict(self) -> dict:
        """Serialize for API/frontend."""
        return {
            'name': self.name,
            'description': self.description,
            'center': list(self.center_world),
            'radius': self.radius_world,
            'shape': self.shape,
            'scale': list(self.scale),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ObstacleConstraint':
        """Deserialize from API/frontend."""
        return cls(
            name=data['name'],
            center_world=tuple(data['center']),
            radius_world=data.get('radius', 1.0),
            shape=data.get('shape', 'ellipse'),
            scale=tuple(data.get('scale', [1.0, 1.0])),
            description=data.get('description', ''),
        )


def create_maze2d_large_obstacles(norm_mins: torch.Tensor, norm_maxs: torch.Tensor) -> List[ControlBarrierFunction]:
    """Create the obstacle barrier functions for the maze2d-large-v1 environment.
    
    Classification: [SPECIFIED] — These obstacles match the upstream code exactly.
    Official repo: diffuser/models/diffusion.py, obstacle definitions in invariance() method.
    
    The upstream code defines 6 obstacles (some with different shapes/scales):
    1. Obstacle 1: center=(5.0, 5.8), radius=1, ellipse (p=2)
    2. Obstacle 2: center=(2.0, 5.3), radius=1, superellipse (p=4)
    3. Obstacle 3: center=(3.0, 2.0), radius=1, y_scale=0.5 superellipse (p=4)
    4. Obstacle 4: center=(3.5, 8.5), radius=1, scale=1.8 superellipse (p=4)
    5. Obstacle 5: center=(7.0, 7.6), radius=1, superellipse (p=4)
    6. Obstacle 6: center=(6.3, 10.0), radius=1, superellipse (p=4)
    """
    obstacles = [
        ObstacleConstraint("obs_1", (5.0, 5.8), 1.0, 'ellipse'),
        ObstacleConstraint("obs_2", (2.0, 5.3), 1.0, 'superellipse'),
        ObstacleConstraint("obs_3", (3.0, 2.0), 1.0, 'superellipse', scale=(0.5, 1.0)),
        ObstacleConstraint("obs_4", (3.5, 8.5), 1.0, 'superellipse', scale=(1.8, 1.8)),
        ObstacleConstraint("obs_5", (7.0, 7.6), 1.0, 'superellipse', margin=0.4),
        ObstacleConstraint("obs_6", (6.3, 10.0), 1.0, 'superellipse'),
    ]
    
    return [obs.to_barrier(norm_mins, norm_maxs) for obs in obstacles]
