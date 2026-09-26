"""
CBF-based Safety Filter for SafeDiffuser denoising.

Classification: [SPECIFIED] — Core contribution of the SafeDiffuser paper.
Paper reference: SafeDiffuser Section 4, Algorithm 1.
Official repo: diffuser/models/diffusion.py, methods invariance(), invariance_time(),
               invariance_cf(), invariance_time_cf()

The safety filter is applied at each denoising step t in the reverse diffusion.
Given:
  - x_t: current noisy trajectory
  - x_{t-1}_proposed: the DDPM-proposed next state
  
The filter solves a Quadratic Program (QP):
  min_u  ||u - u_ref||²
  s.t.   ∇h(x) · u + α(h(x)) ≥ 0   for each obstacle

where:
  - u_ref = x_{t-1}_proposed - x_t  (the proposed update)
  - h(x) is the barrier function for each obstacle
  - ∇h(x) is the barrier gradient (Lie derivative L_g h)
  - α(h) = k·h is the class-K function

The QP finds the minimum-norm correction to u_ref that satisfies all CBF constraints.

Implementation variants from the paper:
  1. RoS (Robust Safety): invariance() — constant safe set
  2. ReS (Relaxed Safety): invariance_relax() — time-decaying relaxation
  3. TVS (Time-Varying Safety): invariance_time() — sigmoid-based time-varying set
  4. Closed-form variants: invariance_cf(), invariance_time_cf() — analytical solutions

We implement a generalized version that supports all variants through configuration.

The upstream uses qpth (differentiable QP solver). We provide both:
  - QP-based solution (requires cvxpy/scipy)
  - Closed-form solution for 1-2 constraints (no external QP solver needed)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from .barrier_functions import ControlBarrierFunction, EllipsoidBarrier, SuperellipseBarrier, TimeVaryingSafetySet


class CBFSafetyFilter(nn.Module):
    """CBF-based safety filter applied during diffusion denoising.
    
    Classification: [SPECIFIED] — Core paper contribution.
    Paper reference: SafeDiffuser Section 4, Algorithm 1.
    
    This module is injected into GaussianDiffusion.p_sample() and modifies 
    the proposed x_{t-1} to satisfy safety constraints.
    
    Args:
        barriers: List of ControlBarrierFunction instances (one per obstacle)
        pos_indices: Indices of position coordinates in the state vector
                     e.g., [2, 3] for maze2d where state = [vel_y, vel_x, pos_y, pos_x]
        method: 'qp' (QP solver), 'closed_form' (analytical), or 'gradient' (projection)
        time_varying: If True, use TVS (time-varying safety sets)
        t_bias: Sigmoid bias for TVS method
        class_k_gain: Gain k in α(h) = k·h
    """
    
    def __init__(
        self,
        barriers: List[ControlBarrierFunction],
        pos_indices: List[int] = [2, 3],
        method: str = 'closed_form',     # [ENGINEERING DECISION] default to closed-form (no QP dep)
        time_varying: bool = True,        # [SPECIFIED] TVS is the best-performing variant
        t_bias: float = 5.0,
        class_k_gain: float = 1.0,
    ):
        super().__init__()
        self.barriers = nn.ModuleList(barriers)
        self.pos_indices = pos_indices
        self.method = method
        self.class_k_gain = class_k_gain
        
        self.time_varying_set = TimeVaryingSafetySet(t_bias) if time_varying else None
        
        # Track safety metrics during inference
        self.last_safety_values = {}

    @torch.no_grad()
    def forward(self, x: torch.Tensor, xp1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply safety filter to the proposed denoising step.
        
        Classification: [SPECIFIED] — This is the SafeDiffuser algorithm.
        
        Args:
            x: Current state x_t [batch, horizon, transition_dim] or [1, batch, horizon, transition_dim]
            xp1: Proposed next state x_{t-1} [same shape]
            t: Current diffusion timestep [batch]
        
        Returns:
            x_safe: Safety-corrected x_{t-1}
        """
        # Handle squeezed/unsqueezed inputs (upstream uses different conventions)
        squeeze = False
        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
            xp1 = xp1.squeeze(0)
            squeeze = True

        batch_size, horizon, dim = xp1.shape
        result = xp1.clone()
        
        # Vectorized batch over all trajectory points
        N = batch_size * horizon
        pos = x[:, :, self.pos_indices].reshape(N, len(self.pos_indices))
        pos_proposed = xp1[:, :, self.pos_indices].reshape(N, len(self.pos_indices))
        ref = pos_proposed - pos

        G_list = []
        h_list = []
        
        for barrier in self.barriers:
            b_val = barrier.h(pos)
            grad_b = barrier.grad_h(pos)
            
            Lfb = 0.0
            if self.time_varying_set is not None:
                t_expanded = t.unsqueeze(1).expand(batch_size, horizon).reshape(-1)
                time_offset = self.time_varying_set.time_offset(t_expanded).view(N, 1)
                Lfb = self.time_varying_set.time_offset_derivative(t_expanded).view(N, 1)
                b_val = b_val - time_offset
            
            G = -grad_b
            h_rhs = Lfb + self.class_k_gain * b_val
            
            G_list.append(G)
            h_list.append(h_rhs)

        if self.method == 'closed_form':
            u_safe = self._closed_form_solve(ref, G_list, h_list)
        elif self.method == 'gradient':
            u_safe = self._gradient_projection(ref, pos, G_list, h_list)
        else:
            u_safe = self._qp_solve(ref, G_list, h_list)
        
        u_safe_reshaped = u_safe.reshape(batch_size, horizon, len(self.pos_indices))
        result[:, :, self.pos_indices] = x[:, :, self.pos_indices] + u_safe_reshaped

        # Track min barrier value for metrics
        self._update_safety_metrics(result, t)
        
        if squeeze:
            result = result.unsqueeze(0)
        return result

    def _closed_form_solve(self, ref: torch.Tensor, G_list: List[torch.Tensor], 
                           h_list: List[torch.Tensor]) -> torch.Tensor:
        """Closed-form CBF-QP solution for 1-2 constraints.
        
        Classification: [SPECIFIED] — Upstream implements this in invariance_cf() and 
        invariance_time_cf() methods.
        
        For a single constraint G·u ≤ h:
            If G·u_ref ≤ h: u = u_ref (already safe)
            Else: u = u_ref - λ·G^T  where λ = max(0, (G·u_ref - h) / (G·G^T))
        
        This is the KKT solution for the QP min ||u - u_ref||² s.t. Gu ≤ h.
        """
        u = ref.clone()
        
        if len(G_list) == 1:
            # Single constraint: analytical solution
            G = G_list[0]  # [batch, 2]
            h = h_list[0]  # [batch, 1]
            
            Gu = (G * u).sum(dim=-1, keepdim=True)  # G · u_ref
            GGt = (G * G).sum(dim=-1, keepdim=True)  # G · G^T
            
            # λ = max(0, (Gu - h) / GGt)
            violation = Gu - h  # positive means constraint violated
            lam = torch.clamp(violation / (GGt + 1e-8), min=0)
            
            u = u - lam * G  # project back to feasible set
            
        elif len(G_list) == 2:
            # Two constraints: analytical solution from upstream invariance_time_cf()
            # Classification: [SPECIFIED] — directly from upstream closed-form code
            G0, G1 = G_list[0], G_list[1]
            h0, h1 = h_list[0], h_list[1]
            
            u_bar = ref  # reference input
            
            # Compute Gram matrix elements
            y1_bar = G0  # [batch, 2]
            y2_bar = G1  # [batch, 2]
            
            p1_bar = h0 - (G0 * u_bar).sum(dim=-1, keepdim=True)
            p2_bar = h1 - (G1 * u_bar).sum(dim=-1, keepdim=True)
            
            # Gram matrix G = [[y1·y1, y1·y2], [y2·y1, y2·y2]]
            g11 = (y1_bar * y1_bar).sum(dim=-1, keepdim=True)
            g12 = (y1_bar * y2_bar).sum(dim=-1, keepdim=True)
            g21 = g12  # symmetric
            g22 = (y2_bar * y2_bar).sum(dim=-1, keepdim=True)
            
            # Clamp p values (feasibility)
            w_p1 = torch.clamp(p1_bar, max=0)
            w_p2 = torch.clamp(p2_bar, max=0)
            
            det = g11 * g22 - g12 * g21 + 1e-8
            
            # Solve 2x2 system with complementarity conditions
            # This matches the upstream logic in invariance_time_cf()
            lambda1 = torch.where(
                g21 * w_p2 < g22 * p1_bar,
                torch.zeros_like(p1_bar),
                torch.where(
                    g12 * w_p1 < g11 * p2_bar,
                    w_p1 / (g11 + 1e-8),
                    torch.clamp(g22 * p1_bar - g21 * p2_bar, max=0) / det
                )
            )
            
            lambda2 = torch.where(
                g21 * w_p2 < g22 * p1_bar,
                w_p2 / (g22 + 1e-8),
                torch.where(
                    g12 * w_p1 < g11 * p2_bar,
                    torch.zeros_like(p2_bar),
                    torch.clamp(g11 * p2_bar - g12 * p1_bar, max=0) / det
                )
            )
            
            u = lambda1 * y1_bar + lambda2 * y2_bar + u_bar

        else:
            # More than 2 constraints: fall back to iterative projection
            u = self._iterative_projection(ref, G_list, h_list)
        
        return u

    def _gradient_projection(self, ref: torch.Tensor, pos: torch.Tensor,
                             G_list: List[torch.Tensor], h_list: List[torch.Tensor]) -> torch.Tensor:
        """Gradient-based projection for safety correction.
        
        Classification: [ENGINEERING DECISION] — Simpler alternative when QP solvers unavailable.
        Iteratively projects the update to satisfy constraints.
        """
        return self._iterative_projection(ref, G_list, h_list, max_iters=10)
    
    def _iterative_projection(self, ref: torch.Tensor, G_list: List[torch.Tensor],
                               h_list: List[torch.Tensor], max_iters: int = 10) -> torch.Tensor:
        """Iterative single-constraint projection (Dykstra-like).
        
        Classification: [ENGINEERING DECISION] — Fallback for >2 constraints.
        """
        u = ref.clone()
        for _ in range(max_iters):
            any_violated = False
            for G, h in zip(G_list, h_list):
                Gu = (G * u).sum(dim=-1, keepdim=True)
                violation = Gu - h
                mask = violation > 0
                if mask.any():
                    any_violated = True
                    GGt = (G * G).sum(dim=-1, keepdim=True) + 1e-8
                    lam = torch.clamp(violation / GGt, min=0)
                    u = u - mask.float() * lam * G
            if not any_violated:
                break
        return u

    def _qp_solve(self, ref: torch.Tensor, G_list: List[torch.Tensor],
                  h_list: List[torch.Tensor]) -> torch.Tensor:
        """Solve CBF-QP using scipy (fallback when qpth unavailable).
        
        Classification: [ADAPTATION FOR LOCAL HARDWARE] — upstream uses qpth PDIPM.
        We use scipy for Windows compatibility.
        
        QP: min_u  0.5 * u^T Q u + q^T u
        s.t. G_ineq u ≤ h_ineq
        
        where Q = I, q = -ref
        """
        try:
            from scipy.optimize import minimize, LinearConstraint
            
            batch_size = ref.shape[0]
            u_out = ref.clone()
            
            for b in range(batch_size):
                ref_b = ref[b].cpu().numpy()
                constraints = []
                
                for G, h in zip(G_list, h_list):
                    g_b = G[b].cpu().numpy()
                    h_b = h[b, 0].cpu().item()
                    # G·u ≤ h  →  -inf ≤ G·u ≤ h
                    constraints.append(LinearConstraint(g_b.reshape(1, -1), -np.inf, h_b))
                
                result = minimize(
                    lambda u: 0.5 * np.sum((u - ref_b) ** 2),
                    ref_b,
                    jac=lambda u: u - ref_b,
                    constraints=constraints,
                    method='SLSQP'
                )
                
                u_out[b] = torch.tensor(result.x, dtype=ref.dtype, device=ref.device)
            
            return u_out
            
        except ImportError:
            # Fall back to closed-form
            return self._closed_form_solve(ref, G_list, h_list)

    def _update_safety_metrics(self, trajectory: torch.Tensor, t: torch.Tensor):
        """Track safety metrics during inference."""
        with torch.no_grad():
            for i, barrier in enumerate(self.barriers):
                pos = trajectory[:, :, self.pos_indices]  # [batch, horizon, 2]
                # Flatten for barrier evaluation
                flat_pos = pos.reshape(-1, 2)
                h_vals = barrier.h(flat_pos)
                min_h = h_vals.min().item()
                self.last_safety_values[f'barrier_{i}_min'] = min_h
                self.last_safety_values[f'barrier_{i}_violations'] = (h_vals < 0).sum().item()

    def get_safety_metrics(self) -> dict:
        """Return latest safety metrics."""
        return dict(self.last_safety_values)
