"""
Trajectory safety validator.

Classification: [ENGINEERING ADDITION] — Additional validation layer beyond the CBF filter.
The paper's safety mechanism operates during denoising; this module provides
post-generation validation and metric computation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from .barrier_functions import ControlBarrierFunction


class TrajectoryValidator:
    """Validates generated trajectories against safety constraints.
    
    This is a post-hoc validator that checks trajectories AFTER generation.
    It complements (does not replace) the CBF safety filter that operates
    DURING denoising.
    
    Classification: [ENGINEERING ADDITION]
    """
    
    def __init__(self, barriers: List[ControlBarrierFunction], pos_indices: List[int] = [2, 3]):
        self.barriers = barriers
        self.pos_indices = pos_indices

    def validate(self, trajectory: torch.Tensor) -> Dict:
        """Validate a trajectory against all barriers.
        
        Args:
            trajectory: [batch, horizon, transition_dim] or [horizon, transition_dim]
            
        Returns:
            dict with validation results and metrics
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
        
        batch_size, horizon, _ = trajectory.shape
        pos = trajectory[:, :, self.pos_indices]  # [batch, horizon, 2]
        
        results = {
            'is_safe': True,
            'total_violations': 0,
            'min_barrier_value': float('inf'),
            'violations_per_obstacle': {},
            'violations_per_timestep': torch.zeros(horizon, dtype=torch.long),
            'barrier_values': {},
        }
        
        for i, barrier in enumerate(self.barriers):
            flat_pos = pos.reshape(-1, 2)
            h_vals = barrier.h(flat_pos)  # [batch*horizon, 1]
            h_vals = h_vals.reshape(batch_size, horizon)
            
            violations = (h_vals < 0)
            n_violations = violations.sum().item()
            min_h = h_vals.min().item()
            
            results['violations_per_obstacle'][f'obstacle_{i}'] = {
                'count': n_violations,
                'min_barrier': min_h,
                'is_violated': n_violations > 0,
            }
            
            results['total_violations'] += n_violations
            results['min_barrier_value'] = min(results['min_barrier_value'], min_h)
            results['violations_per_timestep'] += violations.sum(dim=0).cpu()
            results['barrier_values'][f'obstacle_{i}'] = h_vals.detach().cpu().numpy().tolist()
        
        results['is_safe'] = results['total_violations'] == 0
        total_checks = batch_size * horizon * len(self.barriers)
        results['violation_rate'] = (results['total_violations'] / total_checks) if total_checks > 0 else 0.0
        if results['min_barrier_value'] == float('inf'):
            results['min_barrier_value'] = 0.0
        
        return results

    def compute_metrics(self, trajectory: torch.Tensor, 
                        goal: Optional[torch.Tensor] = None,
                        planning_time: float = 0.0) -> Dict:
        """Compute comprehensive planning metrics.
        
        Args:
            trajectory: [batch, horizon, transition_dim]
            goal: [batch, obs_dim] goal state (optional)
            planning_time: time taken to generate the trajectory
        """
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0)
            
        validation = self.validate(trajectory)
        
        pos = trajectory[:, :, self.pos_indices]  # [batch, horizon, 2]
        
        # Trajectory length
        diffs = pos[:, 1:] - pos[:, :-1]
        step_lengths = torch.norm(diffs, dim=-1)
        total_length = step_lengths.sum(dim=-1).mean().item()
        
        # Goal distance (if provided)
        goal_distance = None
        if goal is not None:
            final_pos = pos[:, -1]  # [batch, 2]
            goal_pos = goal[:, self.pos_indices] if goal.shape[-1] > 2 else goal
            goal_distance = torch.norm(final_pos - goal_pos, dim=-1).mean().item()
        
        metrics = {
            'is_safe': validation['is_safe'],
            'safety_violations': validation['total_violations'],
            'min_safety_margin': validation['min_barrier_value'],
            'violation_rate': validation['violation_rate'],
            'trajectory_length': total_length,
            'planning_time_ms': planning_time * 1000,
            'goal_distance': goal_distance,
            'violations_per_obstacle': validation['violations_per_obstacle'],
        }
        
        return metrics
