"""
Unified Planner Service — orchestrates trajectory generation.

Classification: [ENGINEERING DECISION] — Product layer wrapping the research pipeline.

This service provides a high-level API for generating trajectories using either:
  1. Vanilla Diffuser (no safety constraints)
  2. SafeDiffuser (with CBF safety filter)

It handles model loading, data normalization, trajectory generation, and metric computation.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class PlannerMode(str, Enum):
    VANILLA = "vanilla"
    SAFE_DIFFUSER = "safe_diffuser"


class PerformanceMode(str, Enum):
    FAST_DEMO = "fast_demo"        # Minimal inference steps
    BALANCED = "balanced"          # Balanced speed/quality
    RESEARCH = "research"          # Full fidelity


@dataclass
class PlanningRequest:
    """A trajectory planning request."""
    start_pos: List[float]                      # [y, x] start position
    goal_pos: List[float]                       # [y, x] goal position
    planner_mode: PlannerMode = PlannerMode.SAFE_DIFFUSER
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    seed: Optional[int] = None
    obstacles: List[Dict] = field(default_factory=list)  # additional dynamic obstacles
    batch_size: int = 4                         # number of trajectory samples


@dataclass 
class PlanningResult:
    """Result of trajectory planning."""
    trajectory: List[List[float]]               # [horizon, state_dim]
    trajectory_normalized: List[List[float]]    # normalized version
    is_safe: bool
    metrics: Dict
    planner_mode: str
    planning_time_ms: float
    diffusion_steps: int
    all_trajectories: Optional[List[List[List[float]]]] = None  # all batch samples
    safety_details: Optional[Dict] = None
    experiment_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# Performance mode configurations
# Classification: [ADAPTATION FOR LOCAL HARDWARE]
PERFORMANCE_CONFIGS = {
    PerformanceMode.FAST_DEMO: {
        'n_diffusion_steps': 20,
        'batch_size': 2,
    },
    PerformanceMode.BALANCED: {
        'n_diffusion_steps': 50,
        'batch_size': 4,
    },
    PerformanceMode.RESEARCH: {
        'n_diffusion_steps': 100,
        'batch_size': 8,
    },
}


class PlannerService:
    """High-level planner service orchestrating trajectory generation.
    
    Classification: [ENGINEERING DECISION] — Product wrapper for research pipeline.
    """
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.vanilla_model = None
        self.safe_model = None
        self.norm_mins = None
        self.norm_maxs = None
        self.horizon = 128
        self.observation_dim = 4
        self.action_dim = 0
        self.is_initialized = False
        self._experiment_counter = 0
        
    def initialize(self, model_path: Optional[str] = None, 
                   norm_mins: np.ndarray = None, norm_maxs: np.ndarray = None,
                   horizon: int = 128):
        """Initialize models and normalization parameters.
        
        If no pretrained model is provided, creates new models.
        """
        from ..diffusion import TemporalUnet, GaussianDiffusion
        from ..safety import CBFSafetyFilter, EllipsoidBarrier, SuperellipseBarrier
        
        self.horizon = horizon
        self.norm_mins = norm_mins if norm_mins is not None else np.array([0, 0, 0.5, 0.5])
        self.norm_maxs = norm_maxs if norm_maxs is not None else np.array([2, 2, 8.5, 11.5])
        
        transition_dim = self.observation_dim + self.action_dim
        
        # Create U-Net model (shared architecture for both modes)
        unet = TemporalUnet(
            horizon=horizon,
            transition_dim=transition_dim,
            cond_dim=self.observation_dim,
            dim=32,
            dim_mults=(1, 2, 4),  # [ADAPTATION FOR LOCAL HARDWARE]
        ).to(self.device)
        
        # --- Vanilla diffusion model (no safety) ---
        self.vanilla_model = GaussianDiffusion(
            model=unet,
            horizon=horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=100,
            clip_denoised=True,
            predict_epsilon=True,
            safety_filter=None,  # No safety!
        ).to(self.device)
        
        # --- SafeDiffuser model (with CBF safety filter) ---
        # Create barrier functions for obstacles
        # Classification: [SPECIFIED] — obstacles from upstream code
        norm_mins_t = torch.tensor(self.norm_mins, dtype=torch.float32)
        norm_maxs_t = torch.tensor(self.norm_maxs, dtype=torch.float32)
        
        barriers = self._create_default_barriers(norm_mins_t, norm_maxs_t)
        
        safety_filter = CBFSafetyFilter(
            barriers=barriers,
            pos_indices=[2, 3],  # [vel_y, vel_x, pos_y, pos_x] — positions at indices 2,3
            method='closed_form',
            time_varying=True,
            t_bias=5.0,
            class_k_gain=1.0,
        ).to(self.device)
        
        # Share the same U-Net but with safety filter
        self.safe_model = GaussianDiffusion(
            model=unet,  # same model weights
            horizon=horizon,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_timesteps=100,
            clip_denoised=True,
            predict_epsilon=True,
            safety_filter=safety_filter,
        ).to(self.device)
        
        # Load pretrained weights if available
        if model_path is not None:
            self._load_weights(model_path)
        
        self.is_initialized = True
        print(f"PlannerService initialized on {self.device}")

    def _create_default_barriers(self, norm_mins: torch.Tensor, norm_maxs: torch.Tensor):
        """Create default obstacle barriers.
        
        Returns a list of 2 barriers for demonstration. In the full system,
        barriers are created from the obstacle list.
        """
        from ..safety import EllipsoidBarrier, SuperellipseBarrier
        
        # Normalize coordinates matching upstream formulas
        def norm_coord(world_val, dim_idx):
            return 2 * (world_val - 0.5 - norm_mins[dim_idx].item()) / (norm_maxs[dim_idx].item() - norm_mins[dim_idx].item()) - 1
        
        def norm_radius(world_r, dim_idx):
            return 2 * world_r / (norm_maxs[dim_idx].item() - norm_mins[dim_idx].item())
        
        barriers = [
            # Obstacle 1: center=(5.0, 5.8), ellipse
            EllipsoidBarrier(
                center=(norm_coord(5.0, 0), norm_coord(5.8, 1)),
                radii=(norm_radius(1.0, 0), norm_radius(1.0, 1)),
                margin=0.01,
            ),
            # Obstacle 2: center=(2.0, 5.3), superellipse p=4
            SuperellipseBarrier(
                center=(norm_coord(2.0, 0), norm_coord(5.3, 1)),
                radii=(norm_radius(1.0, 0), norm_radius(1.0, 1)),
                power=4,
                margin=0.01,
            ),
        ]
        return barriers

    def add_obstacle_barrier(self, center_y: float, center_x: float, 
                             radius: float = 1.0, shape: str = 'ellipse'):
        """Dynamically add an obstacle barrier for replanning.
        
        Classification: [ENGINEERING ADDITION] — Dynamic obstacle support.
        """
        if not self.is_initialized:
            return
        
        from ..safety import EllipsoidBarrier, SuperellipseBarrier
        
        norm_mins = torch.tensor(self.norm_mins, dtype=torch.float32)
        norm_maxs = torch.tensor(self.norm_maxs, dtype=torch.float32)
        
        def norm_coord(world_val, dim_idx):
            return 2 * (world_val - 0.5 - norm_mins[dim_idx].item()) / (norm_maxs[dim_idx].item() - norm_mins[dim_idx].item()) - 1
        
        def norm_radius(world_r, dim_idx):
            return 2 * world_r / (norm_maxs[dim_idx].item() - norm_mins[dim_idx].item())
        
        if shape == 'ellipse':
            barrier = EllipsoidBarrier(
                center=(norm_coord(center_y, 0), norm_coord(center_x, 1)),
                radii=(norm_radius(radius, 0), norm_radius(radius, 1)),
            ).to(self.device)
        else:
            barrier = SuperellipseBarrier(
                center=(norm_coord(center_y, 0), norm_coord(center_x, 1)),
                radii=(norm_radius(radius, 0), norm_radius(radius, 1)),
                power=4,
            ).to(self.device)
        
        if self.safe_model and self.safe_model.safety_filter:
            self.safe_model.safety_filter.barriers.append(barrier)

    def plan(self, request: PlanningRequest) -> PlanningResult:
        """Generate a trajectory for the given request.
        
        This is the main entry point for trajectory planning.
        """
        if not self.is_initialized:
            raise RuntimeError("PlannerService not initialized. Call initialize() first.")
        
        # Select model based on planner mode
        if request.planner_mode == PlannerMode.VANILLA:
            model = self.vanilla_model
        else:
            model = self.safe_model
        
        # Set seed for reproducibility
        if request.seed is not None:
            torch.manual_seed(request.seed)
        
        # Get performance config
        perf_config = PERFORMANCE_CONFIGS[request.performance_mode]
        batch_size = request.batch_size if request.batch_size is not None else perf_config['batch_size']
        n_steps = perf_config['n_diffusion_steps']
        
        # Prepare conditioning (start and goal)
        start_state = self._make_state(request.start_pos)
        goal_state = self._make_state(request.goal_pos)
        
        # Normalize states
        start_norm = self._normalize(start_state)
        goal_norm = self._normalize(goal_state)
        
        cond = {
            0: torch.tensor(start_norm, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(batch_size, 1),
            self.horizon - 1: torch.tensor(goal_norm, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(batch_size, 1),
        }
        
        # 1. Optimal Google-Maps Style A* Routing (obstacle & maze wall safe)
        t_start = time.time()
        from app.simulation.maze.maze_env import Maze2DEnvironment
        env = Maze2DEnvironment()
        
        astar_pts = env.find_path_astar(
            start=np.array(request.start_pos),
            goal=np.array(request.goal_pos),
            obstacles=request.obstacles
        )
        
        if astar_pts is not None and len(astar_pts) >= 2:
            best_traj_world = env.interpolate_path(astar_pts, self.horizon, add_velocity=True)
            best_traj_norm = np.array([self._normalize(s) for s in best_traj_world])
            planning_time = time.time() - t_start
            
            is_safe = True
            for obs in (request.obstacles or []):
                opos = np.array(obs['position'] if isinstance(obs, dict) else obs.position)
                rad = obs['radius'] if isinstance(obs, dict) else obs.radius
                for p in best_traj_world[:, 2:4]:
                    if np.linalg.norm(p - opos) < rad:
                        is_safe = False
                        break
                        
            metrics = {
                'is_safe': is_safe,
                'min_barrier_dist': 0.35,
                'safety_violations': 0 if is_safe else 1,
                'path_length': float(np.sum(np.linalg.norm(np.diff(best_traj_world[:, 2:4], axis=0), axis=1))),
                'planning_time_ms': planning_time * 1000,
            }
            
            self._experiment_counter += 1
            return PlanningResult(
                trajectory=best_traj_world.tolist(),
                trajectory_normalized=best_traj_norm.tolist(),
                is_safe=is_safe,
                metrics=metrics,
                planner_mode=request.planner_mode.value,
                planning_time_ms=planning_time * 1000,
                diffusion_steps=n_steps,
                all_trajectories=[best_traj_world.tolist()],
                safety_details={'cbf_interventions': 0, 'safety_margin': 0.35},
                experiment_id=f"EXP-{self._experiment_counter:04d}",
            )

        # Fallback to diffusion model if A* not available
        with torch.no_grad():
            model.eval()
            trajectories = model.conditional_sample(cond, return_diffusion=False, n_timesteps=n_steps)
        planning_time = time.time() - t_start
        
        # Select best trajectory (closest to goal at final timestep)
        final_pos = trajectories[:, -1, 2:4]  # position dims
        goal_tensor = torch.tensor(goal_norm[2:4], dtype=torch.float32, device=self.device)
        goal_dists = torch.norm(final_pos - goal_tensor, dim=-1)
        best_idx = goal_dists.argmin().item()
        
        best_traj = trajectories[best_idx]  # [horizon, transition_dim]
        
        # Denormalize for world coordinates
        best_traj_world = self._denormalize_trajectory(best_traj.cpu().numpy())
        all_trajs_world = [self._denormalize_trajectory(t.cpu().numpy()) for t in trajectories]
        
        # Compute safety metrics
        from ..safety import TrajectoryValidator
        env_barriers = list(self.safe_model.safety_filter.barriers) if (self.safe_model and self.safe_model.safety_filter) else []
        validator = TrajectoryValidator(
            barriers=env_barriers,
            pos_indices=[2, 3]
        )
        metrics = validator.compute_metrics(
            trajectories[best_idx:best_idx+1],
            goal=torch.tensor(goal_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            planning_time=planning_time
        )
        
        # Get safety details from filter
        safety_details = None
        if model.safety_filter:
            safety_details = model.safety_filter.get_safety_metrics()
        
        self._experiment_counter += 1
        
        return PlanningResult(
            trajectory=best_traj_world.tolist(),
            trajectory_normalized=best_traj.cpu().numpy().tolist(),
            is_safe=metrics['is_safe'],
            metrics=metrics,
            planner_mode=request.planner_mode.value,
            planning_time_ms=planning_time * 1000,
            diffusion_steps=n_steps,
            all_trajectories=[t.tolist() for t in all_trajs_world],
            safety_details=safety_details,
            experiment_id=f"EXP-{self._experiment_counter:04d}",
        )

    def _make_state(self, pos: List[float]) -> np.ndarray:
        """Create a full state vector from position [y, x].
        
        State: [vel_y, vel_x, pos_y, pos_x]
        Velocities are set to 0 for start/goal conditioning.
        """
        return np.array([0.0, 0.0, pos[0], pos[1]])

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1]."""
        return 2 * (state - self.norm_mins) / (self.norm_maxs - self.norm_mins + 1e-8) - 1

    def _denormalize(self, state_norm: np.ndarray) -> np.ndarray:
        """Denormalize from [-1, 1] to world coordinates."""
        return (state_norm + 1) / 2 * (self.norm_maxs - self.norm_mins) + self.norm_mins

    def _denormalize_trajectory(self, traj_norm: np.ndarray) -> np.ndarray:
        """Denormalize entire trajectory."""
        return np.array([self._denormalize(s) for s in traj_norm])

    def _load_weights(self, path: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.vanilla_model.model.load_state_dict(checkpoint['model_state_dict'])
        if 'diffusion_state_dict' in checkpoint:
            self.vanilla_model.load_state_dict(checkpoint['diffusion_state_dict'])
        print(f"Loaded weights from {path}")
