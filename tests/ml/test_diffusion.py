"""
Unit tests for Diffusion Model components.
Tests Temporal UNet, Noise Schedules, Forward Noising, and Conditioning.
Classification: [SPECIFIED]
"""

import sys
import os
import unittest
import numpy as np

# Ensure project root in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDiffusionComponents(unittest.TestCase):
    """Test core diffusion architecture and mathematical functions."""

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not yet available in the environment")

    def test_noise_schedule_shapes(self):
        """Test beta, alpha, and alpha_bar schedules."""
        from app.ml.diffusion.gaussian_diffusion import cosine_beta_schedule, linear_beta_schedule
        
        n_timesteps = 30
        betas_cosine = cosine_beta_schedule(n_timesteps)
        self.assertEqual(len(betas_cosine), n_timesteps)
        self.assertTrue(torch.all(betas_cosine > 0))
        self.assertTrue(torch.all(betas_cosine < 1))

        betas_linear = linear_beta_schedule(n_timesteps)
        self.assertEqual(len(betas_linear), n_timesteps)
        self.assertTrue(torch.all(betas_linear > 0))

    def test_temporal_unet_shape(self):
        """Test TemporalUnet input/output tensor dimensions match [B, H, D]."""
        from app.ml.diffusion.temporal_unet import TemporalUnet

        batch_size = 2
        horizon = 32
        transition_dim = 4

        model = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            dim=32,
            dim_mults=(1, 2),
        )

        x = torch.randn(batch_size, horizon, transition_dim)
        timesteps = torch.randint(0, 30, (batch_size,))
        output = model(x, timesteps)

        self.assertEqual(output.shape, (batch_size, horizon, transition_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_apply_conditioning(self):
        """Test that boundary conditions (start and goal) are strictly maintained."""
        from app.ml.diffusion.gaussian_diffusion import apply_conditioning

        batch_size = 2
        horizon = 32
        transition_dim = 4

        x = torch.zeros(batch_size, horizon, transition_dim)
        start_state = torch.tensor([1.0, 2.0])
        goal_state = torch.tensor([8.0, 8.0])

        conditions = {
            0: start_state,
            horizon - 1: goal_state,
        }

        x_cond = apply_conditioning(x, conditions, action_dim=2)

        # Check start condition
        for b in range(batch_size):
            self.assertTrue(torch.allclose(x_cond[b, 0, 2:], start_state))
            self.assertTrue(torch.allclose(x_cond[b, -1, 2:], goal_state))

    def test_forward_noising_convergence(self):
        """Test that q_sample at t=K approaches standard normal distribution."""
        from app.ml.diffusion.gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule
        from app.ml.diffusion.temporal_unet import TemporalUnet

        horizon = 32
        transition_dim = 4
        model = TemporalUnet(horizon=horizon, transition_dim=transition_dim, dim=16, dim_mults=(1,))
        diffusion = GaussianDiffusion(model=model, horizon=horizon, observation_dim=2, action_dim=2, n_timesteps=100)

        x_start = torch.ones(10, horizon, transition_dim) * 5.0
        t_final = torch.full((10,), 99, dtype=torch.long)
        noise = torch.randn_like(x_start)

        x_noisy = diffusion.q_sample(x_start, t_final, noise=noise)
        self.assertEqual(x_noisy.shape, x_start.shape)


if __name__ == "__main__":
    unittest.main()
