"""
Gaussian Diffusion process for trajectory planning.

Adapted from the official SafeDiffuser repository (Weixy21/SafeDiffuser).
This implements both the forward diffusion (noising) and reverse diffusion (denoising)
processes for trajectory generation.

Classification: [SPECIFIED] — Core diffusion framework from the paper.
Paper reference: SafeDiffuser Section 3 — Equations 1-5 (DDPM diffusion process)
Official repo: diffuser/models/diffusion.py, class GaussianDiffusion

The safety mechanism (SafeDiffuser's core contribution) is integrated here via
the `safety_filter` parameter, which is applied during denoising (p_sample).
The safety filter implements Control Barrier Functions (CBFs) as described in
SafeDiffuser Section 4.

Local adaptations:
  - [ADAPTATION FOR LOCAL HARDWARE] Default n_timesteps=100 (paper uses 200/256)
  - [ENGINEERING DECISION] Safety filter is injected as a composable module
  - [ENGINEERING DECISION] Removed upstream-specific norm_mins/norm_maxs storage
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule as proposed in Nichol & Dhariwal (2021).
    
    Classification: [SPECIFIED] — Standard cosine schedule used in Diffuser/SafeDiffuser.
    Paper reference: Referenced in SafeDiffuser implementation.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=torch.float32)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule as in original DDPM (Ho et al., 2020).
    
    Classification: [SPECIFIED] — Standard linear schedule.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def extract(a, t, x_shape):
    """Extract values from `a` at indices `t`, reshaped for broadcasting."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def apply_conditioning(x, conditions, action_dim):
    """Apply start/goal conditioning to trajectory.
    
    Classification: [SPECIFIED] — Conditioning mechanism from Diffuser.
    The conditions dict maps timestep index → observation values.
    E.g., conditions = {0: start_obs, -1: goal_obs}
    """
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class WeightedLoss(nn.Module):
    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'a0_loss': a0_loss}


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


LOSS_FUNCTIONS = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


# ---------------------------------------------------------------------------
# Main Gaussian Diffusion model
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion model for trajectory planning.
    
    Implements DDPM-style forward and reverse diffusion on trajectory sequences.
    During inference, optionally applies a safety filter (SafeDiffuser's CBF mechanism)
    at each denoising step.
    
    Classification: [SPECIFIED] — Core model from the paper.
    Paper reference: SafeDiffuser Sections 3-4.
    Official repo: diffuser/models/diffusion.py, class GaussianDiffusion
    
    Args:
        model: The denoising neural network (TemporalUnet)
        horizon: Number of timesteps in the generated trajectory
        observation_dim: Dimension of observation space
        action_dim: Dimension of action space (0 for state-only trajectories like Maze2D)
        n_timesteps: Number of diffusion timesteps
        loss_type: 'l1' or 'l2'
        clip_denoised: Whether to clip x_0 predictions to [-1, 1]
        predict_epsilon: If True, model predicts noise; else predicts x_0
        safety_filter: Optional SafeDiffuser safety module applied during denoising
    """

    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=100,           # [ADAPTATION FOR LOCAL HARDWARE] paper uses 200/256
        loss_type='l2',
        clip_denoised=True,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        safety_filter=None,        # [ENGINEERING DECISION] composable safety module
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.safety_filter = safety_filter

        # --- Noise schedule ---
        # Classification: [SPECIFIED] — Cosine schedule as in the official repo
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Forward diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        # Classification: [SPECIFIED] — Standard DDPM posterior (Eq. 7 in Ho et al. 2020)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        # Loss
        loss_weights_tensor = self._get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = LOSS_FUNCTIONS[loss_type](loss_weights_tensor, self.action_dim)

        # Safety tracking
        self.safety_values = []  # populated during inference

    def _get_loss_weights(self, action_weight, discount, weights_dict):
        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        if self.action_dim > 0:
            loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    # ======================================================================
    # Sampling (inference)
    # ======================================================================

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise.
        
        Classification: [SPECIFIED] — Standard DDPM x_0 prediction.
        x_0 = (1/√ᾱ_t) * x_t - (√(1/ᾱ_t - 1)) * ε
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0).
        
        Classification: [SPECIFIED] — Standard DDPM posterior.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, cond, t):
        """Compute p(x_{t-1} | x_t) — the learned reverse distribution.
        
        Classification: [SPECIFIED] — Standard reverse step.
        """
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model(x, cond, t)
        )
        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        """Single reverse diffusion step with optional safety filtering.
        
        Classification: [SPECIFIED] — This is where SafeDiffuser's core contribution lives.
        Paper reference: SafeDiffuser Section 4, Algorithm 1.
        
        The standard DDPM step produces x_{t-1} from x_t. SafeDiffuser modifies 
        this by applying a safety filter (CBF-based QP or closed-form correction)
        to the proposed x_{t-1}, ensuring it remains in the safe set.
        
        Official repo: diffuser/models/diffusion.py, p_sample() lines 1030-1097
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Standard DDPM sample
        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # --- SafeDiffuser safety filter ---
        # Classification: [SPECIFIED] — Core paper contribution
        # This is the key integration point. The safety_filter modifies xp1 to satisfy
        # the CBF constraint: ḣ(x_t) ≥ -α(h(x_t))
        if self.safety_filter is not None:
            x_safe = self.safety_filter(x, xp1, t)
            return x_safe
        else:
            return xp1

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=False, return_diffusion=False, n_timesteps=None):
        """Full reverse diffusion loop: x_T → x_0.
        
        Classification: [SPECIFIED] — Standard DDPM sampling loop with optional sub-sampling.
        Official repo: diffuser/models/diffusion.py, p_sample_loop()
        """
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion:
            diffusion = [x.clone()]

        self.safety_values = []

        total_steps = n_timesteps if n_timesteps is not None else self.n_timesteps
        total_steps = min(total_steps, self.n_timesteps)
        if total_steps < self.n_timesteps:
            step_indices = np.linspace(0, self.n_timesteps - 1, total_steps, dtype=int)
            step_indices = sorted(list(set(step_indices.tolist())))
        else:
            step_indices = list(range(self.n_timesteps))

        for i in reversed(step_indices):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)

            if return_diffusion:
                diffusion.append(x.clone())

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, return_diffusion=False, n_timesteps=None, **kwargs):
        """Generate trajectories conditioned on start/goal.
        
        Args:
            cond: dict mapping timestep → observation tensor
                  e.g., {0: start_obs, horizon-1: goal_obs}
            horizon: override trajectory length
            return_diffusion: if True, also return intermediate denoising steps
            n_timesteps: optional sub-sampled step count for fast inference
        """
        device = self.betas.device
        batch_size = len(list(cond.values())[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, return_diffusion=return_diffusion, n_timesteps=n_timesteps, **kwargs)

    # ======================================================================
    # Training
    # ======================================================================

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to x_0 to get x_t.
        
        Classification: [SPECIFIED] — Standard DDPM forward process.
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t):
        """Compute training loss: predict noise from noisy trajectory.
        
        Classification: [SPECIFIED] — Standard DDPM training objective.
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        return loss, info

    def loss(self, x, cond):
        """Compute loss for a batch of training data."""
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
