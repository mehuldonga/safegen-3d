"""
Control Barrier Functions (CBFs) for SafeDiffuser.

Classification: [SPECIFIED] — Core theoretical contribution of the paper.
Paper reference: SafeDiffuser Section 4.1 — "Safety with Control Barrier Functions"

A Control Barrier Function h(x) defines a safe set C = {x : h(x) ≥ 0}.
The CBF condition for forward invariance is:
    ḣ(x, u) ≥ -α(h(x))
where α is a class-K function (typically α(h) = k·h for linear class-K).

In the SafeDiffuser context:
  - x is the current trajectory state at a denoising step
  - u is the proposed update (xp1 - x) from the diffusion model
  - h(x) measures how "safe" the state is (positive = safe, negative = unsafe)
  - The CBF constraint modifies u to ensure the trajectory stays safe

Official repo: diffuser/models/diffusion.py — methods invariance(), invariance_time(), etc.

The upstream code uses several barrier function shapes:
  - Circular/ellipsoidal: h = ((y-oy)/r_y)^2 + ((x-ox)/r_x)^2 - 1  (p=2)
  - Superelliptical:      h = ((y-oy)/r_y)^p + ((x-ox)/r_x)^p - 1  (p=4)

We generalize these into reusable barrier function classes.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ControlBarrierFunction(nn.Module):
    """Base class for Control Barrier Functions.
    
    Classification: [SPECIFIED] — CBF concept from Section 4.1.
    
    A CBF h(x) defines:
      - Safe set: C = {x : h(x) ≥ 0}
      - Unsafe region: h(x) < 0
    
    Subclasses implement h(x) and its gradient ∇h(x) for specific constraint geometries.
    """
    
    def __init__(self, class_k_gain: float = 1.0, margin: float = 0.01):
        """
        Args:
            class_k_gain: k in α(h) = k·h (linear class-K function)
                          Classification: [PARTIALLY SPECIFIED] — paper uses k=1 in code
            margin: safety margin ε subtracted from h
                    Classification: [PARTIALLY SPECIFIED] — upstream uses 0.01
        """
        super().__init__()
        self.class_k_gain = class_k_gain
        self.margin = margin

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate barrier function h(x). Positive = safe."""
        raise NotImplementedError

    def grad_h(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of h w.r.t. the position coordinates."""
        raise NotImplementedError

    def is_safe(self, x: torch.Tensor) -> torch.Tensor:
        """Check if states are in the safe set. Returns bool tensor."""
        return self.h(x) >= 0


class EllipsoidBarrier(ControlBarrierFunction):
    """Ellipsoidal CBF: h = (Δy/r_y)² + (Δx/r_x)² - 1 - ε
    
    Classification: [SPECIFIED] — Used for obstacle 1 in upstream code.
    Official repo: diffuser/models/diffusion.py, invariance() method.
    
    This defines an ellipsoidal obstacle. Points inside the ellipsoid have h < 0 (unsafe).
    Points outside have h > 0 (safe).
    
    Args:
        center: (cy, cx) center of the obstacle in normalized coordinates
        radii: (ry, rx) radii of the ellipsoid in normalized coordinates
    """
    
    def __init__(self, center: Tuple[float, float], radii: Tuple[float, float],
                 class_k_gain: float = 1.0, margin: float = 0.01):
        super().__init__(class_k_gain, margin)
        self.register_buffer('center', torch.tensor(center, dtype=torch.float32))
        self.register_buffer('radii', torch.tensor(radii, dtype=torch.float32))

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 2] tensor of (y, x) positions.
        Returns: [..., 1] barrier values.
        """
        delta = x - self.center
        normalized = delta / self.radii
        return (normalized ** 2).sum(dim=-1, keepdim=True) - 1.0 - self.margin

    def grad_h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of h w.r.t. x.
        ∂h/∂x_i = 2 * (x_i - c_i) / r_i²
        
        Returns: [..., 2] gradient tensor.
        """
        delta = x - self.center
        return 2.0 * delta / (self.radii ** 2)


class SuperellipseBarrier(ControlBarrierFunction):
    """Superellipse CBF: h = (Δy/r_y)^p + (Δx/r_x)^p - 1 - ε
    
    Classification: [SPECIFIED] — Used for obstacles 2-6 in upstream code (p=4).
    Official repo: diffuser/models/diffusion.py, invariance() method, obstacle blocks.
    
    The upstream code uses p=4 for sharper, more "boxy" constraint boundaries.
    
    Args:
        center: (cy, cx) center of the obstacle in normalized coordinates
        radii: (ry, rx) radii in normalized coordinates
        power: exponent p (upstream uses p=4)
    """
    
    def __init__(self, center: Tuple[float, float], radii: Tuple[float, float],
                 power: int = 4, class_k_gain: float = 1.0, margin: float = 0.01):
        super().__init__(class_k_gain, margin)
        self.power = power
        self.register_buffer('center', torch.tensor(center, dtype=torch.float32))
        self.register_buffer('radii', torch.tensor(radii, dtype=torch.float32))

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 2] tensor of (y, x) positions.
        Returns: [..., 1] barrier values.
        """
        delta = x - self.center
        normalized = delta / self.radii
        return (normalized.abs() ** self.power).sum(dim=-1, keepdim=True) - 1.0 - self.margin

    def grad_h(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient of h w.r.t. x.
        ∂h/∂x_i = p * ((x_i - c_i) / r_i)^(p-1) * sign(x_i - c_i) / r_i
                 = p * ((x_i - c_i) / r_i)^(p-1) / r_i   (for even p)
        
        Returns: [..., 2] gradient tensor.
        """
        delta = x - self.center
        normalized = delta / self.radii
        # For even power, sign is handled by the power itself
        return self.power * (normalized ** (self.power - 1)) / self.radii


class TimeVaryingSafetySet(nn.Module):
    """Time-varying safety set for TVS-SafeDiffuser.
    
    Classification: [SPECIFIED] — Paper Section 4.2, "Time-Varying Safe Sets"
    Paper reference: SafeDiffuser Eq. (11) — finite-time diffusion invariance.
    Official repo: diffuser/models/diffusion.py, invariance_time() method.
    
    In TVS-SafeDiffuser, the safe set radius varies with diffusion time:
        h_t(x) = (Δx/r)^p - σ(t_bias - t) - ε
    
    where σ is the sigmoid function. This allows the safe set to be larger
    at early (noisy) diffusion steps and tighter at later (cleaner) steps.
    
    This is the key insight of "finite-time diffusion invariance": safety 
    constraints don't need to be as tight when the trajectory is still very 
    noisy, because the noise will be removed in subsequent steps.
    """
    
    def __init__(self, t_bias: float = 5.0):
        """
        Args:
            t_bias: sigmoid shift parameter.
                    Classification: [PARTIALLY SPECIFIED] — upstream uses t_bias=5
                    with comment "50" suggesting it was tuned from 50.
        """
        super().__init__()
        self.t_bias = t_bias

    def time_offset(self, t: torch.Tensor) -> torch.Tensor:
        """σ(t_bias - t) — the time-varying offset for the barrier function."""
        return torch.sigmoid(self.t_bias - t.float())

    def time_offset_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt σ(t_bias - t) = σ(t_bias - t) * (1 - σ(t_bias - t))
        
        This is L_f h in the CBF constraint, representing how the safe set 
        boundary itself is changing over time.
        
        Classification: [SPECIFIED] — Appears in upstream as Lfb.
        """
        sig = self.time_offset(t)
        return sig * (1 - sig)
