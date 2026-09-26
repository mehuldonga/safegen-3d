"""
Unit tests for Safety Filter and Barrier Functions.
Tests EllipsoidBarrier, SuperellipseBarrier, CBFSafetyFilter, and TrajectoryValidator.
Classification: [SPECIFIED]
"""

import sys
import os
import unittest
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from app.ml.safety.barrier_functions import EllipsoidBarrier, SuperellipseBarrier
from app.ml.safety.cbf_filter import CBFSafetyFilter
from app.ml.safety.validators import TrajectoryValidator


class TestSafetyComponents(unittest.TestCase):
    """Test barrier functions and safety projection mechanisms."""

    def test_ellipsoid_barrier_signs(self):
        """Test that barrier h(x) > 0 outside obstacle, h(x) < 0 inside, and h(x) = 0 on boundary."""
        # Obstacle centered at (4.0, 4.0) with radii (1.0, 1.0), margin=0
        barrier = EllipsoidBarrier(center=(4.0, 4.0), radii=(1.0, 1.0), margin=0.0)

        # Center: deep inside -> h = -1.0
        pt_center = torch.tensor([[4.0, 4.0]])
        h_center = barrier.h(pt_center)
        self.assertAlmostEqual(h_center.item(), -1.0, places=3)
        self.assertTrue(h_center.item() < 0)

        # Boundary: (4.0, 5.0) -> h = 0.0
        pt_boundary = torch.tensor([[4.0, 5.0]])
        h_boundary = barrier.h(pt_boundary)
        self.assertAlmostEqual(h_boundary.item(), 0.0, places=3)

        # Outside: (4.0, 7.0) -> h > 0
        pt_outside = torch.tensor([[4.0, 7.0]])
        h_outside = barrier.h(pt_outside)
        self.assertTrue(h_outside.item() > 0)

    def test_barrier_gradient_direction(self):
        """Test that spatial gradient ∇_x h points outwards away from obstacle center."""
        barrier = EllipsoidBarrier(center=(3.0, 3.0), radii=(1.0, 1.0))
        # Point to the right of center: (3.0, 4.0)
        pt = torch.tensor([[3.0, 4.0]])
        grad = barrier.grad_h(pt)

        # grad w.r.t (y, x): dy should be 0, dx should be positive
        self.assertAlmostEqual(grad[0, 0].item(), 0.0, places=3)
        self.assertTrue(grad[0, 1].item() > 0)

    def test_closed_form_cbf_filter(self):
        """Test closed-form CBF filter modifies unsafe updates."""
        barrier = EllipsoidBarrier(center=(4.0, 4.0), radii=(1.0, 1.0))
        safety_filter = CBFSafetyFilter(barriers=[barrier], pos_indices=[2, 3], method='closed_form')

        # Current state just outside obstacle at (4.0, 5.2)
        x = torch.zeros(1, 10, 4)
        x[:, :, 2] = 4.0
        x[:, :, 3] = 5.2

        # Proposed next step moves INTO obstacle (towards x=4.0)
        xp1 = x.clone()
        xp1[:, :, 3] = 4.5  # penetrates barrier

        t = torch.zeros(1, dtype=torch.long)
        x_filtered = safety_filter(x, xp1, t)

        # Filtered position should not penetrate as deeply as xp1
        self.assertEqual(x_filtered.shape, xp1.shape)
        self.assertTrue(x_filtered[0, 0, 3].item() > xp1[0, 0, 3].item())

    def test_trajectory_validator_metrics(self):
        """Test trajectory validator correctly computes safety margin and collision flag."""
        barrier = EllipsoidBarrier(center=(5.0, 5.0), radii=(1.0, 1.0), margin=0.0)
        validator = TrajectoryValidator(barriers=[barrier], pos_indices=[0, 1])

        # Safe trajectory: line from (0,0) to (0,10)
        safe_traj = torch.zeros(20, 2)
        safe_traj[:, 1] = torch.linspace(0, 10, 20)
        results_safe = validator.validate(safe_traj)
        self.assertTrue(results_safe['is_safe'])
        self.assertEqual(results_safe['total_violations'], 0)
        self.assertTrue(results_safe['min_barrier_value'] > 0)

        # Unsafe trajectory: line from (5,0) to (5,10) cutting directly through (5,5)
        unsafe_traj = torch.zeros(20, 2)
        unsafe_traj[:, 0] = 5.0
        unsafe_traj[:, 1] = torch.linspace(0, 10, 20)
        results_unsafe = validator.validate(unsafe_traj)
        self.assertFalse(results_unsafe['is_safe'])
        self.assertTrue(results_unsafe['total_violations'] > 0)
        self.assertTrue(results_unsafe['min_barrier_value'] < 0)


if __name__ == "__main__":
    unittest.main()
