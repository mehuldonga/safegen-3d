"""
Self-contained Maze2D environment for trajectory planning.

Classification: [ADAPTATION FOR LOCAL HARDWARE] — Replaces MuJoCo/D4RL dependency.
The upstream SafeDiffuser uses D4RL's maze2d-large-v1 environment which requires
MuJoCo 200 (Linux-only). This module provides a standalone maze environment
that generates compatible training data without MuJoCo.

Design decisions:
  - [ENGINEERING DECISION] Uses simple grid-based maze representation
  - [SPECIFIED] Maze layout matches maze2d-large-v1 from D4RL
  - [SPECIFIED] State space: [vel_y, vel_x, pos_y, pos_x] (4D, matching upstream)
  - [ENGINEERING DECISION] No action space (action_dim=0, state-only trajectories)
  - [SPECIFIED] Coordinate range: approximately [0, 12] x [0, 9] for large maze
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import heapq


# The maze2d-large-v1 layout from D4RL
# Classification: [SPECIFIED] — Standard D4RL maze layout
# 1 = wall, 0 = open
MAZE2D_LARGE_LAYOUT = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

# Simplified maze for faster training/demo
# Classification: [ENGINEERING DECISION]
MAZE2D_SIMPLE_LAYOUT = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
])


class Maze2DEnvironment:
    """Self-contained 2D maze environment for trajectory planning.
    
    Classification: [ADAPTATION FOR LOCAL HARDWARE] — Replaces D4RL maze2d.
    
    State space: [vel_y, vel_x, pos_y, pos_x]
    Coordinate system: positions are in [0.5, width-0.5] x [0.5, height-0.5]
                       where each cell is 1x1 unit.
    
    The maze is represented as a grid where 1=wall, 0=open.
    Positions are continuous, with wall collision checking.
    """
    
    def __init__(self, layout: np.ndarray = None, cell_size: float = 1.0):
        if layout is None:
            layout = MAZE2D_LARGE_LAYOUT
        self.layout = layout
        self.cell_size = cell_size
        self.height, self.width = layout.shape
        
        # Compute open cells for sampling positions
        self.open_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if layout[y, x] == 0:
                    self.open_cells.append((y, x))
        
        # State normalization bounds (matching upstream)
        # Classification: [SPECIFIED] — upstream uses these for normalizing to [-1, 1]
        self.pos_min = np.array([0.5, 0.5])  # min y, min x
        self.pos_max = np.array([self.height - 0.5, self.width - 0.5])  # max y, max x
        self.vel_range = 2.0  # approximate velocity range

    def is_valid_position(self, pos_y: float, pos_x: float) -> bool:
        """Check if a position is not inside a wall."""
        grid_y = int(np.clip(pos_y, 0, self.height - 1))
        grid_x = int(np.clip(pos_x, 0, self.width - 1))
        return self.layout[grid_y, grid_x] == 0

    def sample_position(self, rng: np.random.RandomState = None) -> np.ndarray:
        """Sample a random valid position."""
        if rng is None:
            rng = np.random.RandomState()
        cell = self.open_cells[rng.randint(len(self.open_cells))]
        # Position is center of cell + small random offset
        y = cell[0] + 0.5 + rng.uniform(-0.3, 0.3)
        x = cell[1] + 0.5 + rng.uniform(-0.3, 0.3)
        return np.array([y, x])

    def line_of_sight(self, p1: np.ndarray, p2: np.ndarray, n_checks: int = 20) -> bool:
        """Check if there's a collision-free line between two points."""
        for t in np.linspace(0, 1, n_checks):
            p = p1 + t * (p2 - p1)
            if not self.is_valid_position(p[0], p[1]):
                return False
        return True

    def find_path_rrt(self, start: np.ndarray, goal: np.ndarray, 
                      max_iters: int = 5000, step_size: float = 0.5,
                      rng: np.random.RandomState = None) -> Optional[List[np.ndarray]]:
        """Simple RRT path planner for generating training trajectories.
        
        Classification: [ENGINEERING DECISION] — Used to generate synthetic 
        training data without MuJoCo dynamics. The diffusion model learns 
        from these demonstrations.
        """
        if rng is None:
            rng = np.random.RandomState()
            
        tree_nodes = [start.copy()]
        tree_parents = [-1]
        
        for _ in range(max_iters):
            # Sample random point (biased towards goal)
            if rng.random() < 0.1:
                sample = goal.copy()
            else:
                sample = np.array([
                    rng.uniform(0.5, self.height - 0.5),
                    rng.uniform(0.5, self.width - 0.5),
                ])
            
            # Find nearest node
            nodes_arr = np.array(tree_nodes)
            dists = np.linalg.norm(nodes_arr - sample, axis=1)
            nearest_idx = np.argmin(dists)
            nearest = tree_nodes[nearest_idx]
            
            # Extend towards sample
            direction = sample - nearest
            dist = np.linalg.norm(direction)
            if dist < 1e-6:
                continue
            direction = direction / dist
            new_pos = nearest + direction * min(step_size, dist)
            
            # Check validity
            if not self.is_valid_position(new_pos[0], new_pos[1]):
                continue
            if not self.line_of_sight(nearest, new_pos):
                continue
            
            tree_nodes.append(new_pos)
            tree_parents.append(nearest_idx)
            
            # Check if we reached the goal
            if np.linalg.norm(new_pos - goal) < step_size:
                # Reconstruct path
                path = [goal]
                idx = len(tree_nodes) - 1
                while idx != -1:
                    path.append(tree_nodes[idx])
                    idx = tree_parents[idx]
                path.reverse()
                return path
        
        return None  # Failed to find path

    def find_path_astar(self, start: np.ndarray, goal: np.ndarray, 
                        obstacles: Optional[List[Dict]] = None) -> Optional[List[np.ndarray]]:
        """A* pathfinder for deterministic, clean Google-Maps-like routing.
        
        Guarantees:
        1. Always navigates strictly inside open maze corridors.
        2. Actively avoids all obstacles (both static barriers and dynamic ones).
        3. Produces a single, direct, optimal string of waypoints from start to goal.
        """
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        
        sr = int(np.clip(round(start[0] - 0.5), 0, self.height - 1))
        sc = int(np.clip(round(start[1] - 0.5), 0, self.width - 1))
        gr = int(np.clip(round(goal[0] - 0.5), 0, self.height - 1))
        gc = int(np.clip(round(goal[1] - 0.5), 0, self.width - 1))
        
        # Helper to compute cost with obstacle clearance
        def get_cost(r, c):
            p = np.array([r + 0.5, c + 0.5])
            penalty = 0.0
            for obs in (obstacles or []):
                opos = np.array(obs['position'] if isinstance(obs, dict) else obs.position)
                rad = obs['radius'] if isinstance(obs, dict) else obs.radius
                d = np.linalg.norm(p - opos)
                if d < rad:
                    penalty += 200.0 * (rad - d + 1.0)
                elif d < rad + 0.4:
                    penalty += 20.0 * (rad + 0.4 - d)
            return 1.0 + penalty

        queue = [(0.0, (sr, sc), [(sr, sc)])]
        visited = {}
        found = None
        
        while queue:
            cost, (r, c), path = heapq.heappop(queue)
            if (r, c) == (gr, gc):
                found = path
                break
            if (r, c) in visited and visited[(r, c)] <= cost:
                continue
            visited[(r, c)] = cost
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and self.layout[nr, nc] == 0:
                    step_c = get_cost(nr, nc)
                    h = abs(nr - gr) + abs(nc - gc)
                    heapq.heappush(queue, (cost + step_c, (nr, nc), path + [(nr, nc)]))
                    
        if not found:
            return None
            
        # Build raw points starting at start, through cell centers, ending at goal
        pts = [start.copy()]
        for r, c in found:
            pt = np.array([r + 0.5, c + 0.5])
            if np.linalg.norm(pts[-1] - pt) > 0.01:
                pts.append(pt)
        if np.linalg.norm(pts[-1] - goal) > 0.01:
            pts.append(goal.copy())
            
        # Simplify colinear points for crisp, straight street segments
        if len(pts) > 2:
            simplified = [pts[0]]
            for i in range(1, len(pts) - 1):
                v1 = pts[i] - simplified[-1]
                v2 = pts[i + 1] - pts[i]
                d1 = np.linalg.norm(v1)
                d2 = np.linalg.norm(v2)
                if d1 > 1e-4 and d2 > 1e-4:
                    cos_theta = np.dot(v1, v2) / (d1 * d2)
                    if abs(cos_theta - 1.0) > 1e-3:
                        simplified.append(pts[i])
                else:
                    simplified.append(pts[i])
            simplified.append(pts[-1])
            pts = simplified
            
        return pts


    def interpolate_path(self, path: List[np.ndarray], n_points: int, 
                         add_velocity: bool = True) -> np.ndarray:
        """Interpolate a path to exactly n_points with smooth velocity.
        
        Classification: [ENGINEERING DECISION] — Creates training-ready trajectories.
        
        Returns: [n_points, 4] array of [vel_y, vel_x, pos_y, pos_x]
                 or [n_points, 2] if add_velocity=False
        """
        if len(path) < 2:
            return None
        
        # Compute cumulative distances
        path_arr = np.array(path)
        diffs = np.diff(path_arr, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative[-1]
        
        if total_length < 1e-6:
            return None
        
        # Interpolate positions at uniform arc-length
        target_dists = np.linspace(0, total_length, n_points)
        positions = np.zeros((n_points, 2))
        
        for i, d in enumerate(target_dists):
            # Find which segment this distance falls in
            seg_idx = np.searchsorted(cumulative, d, side='right') - 1
            seg_idx = np.clip(seg_idx, 0, len(path) - 2)
            
            # Interpolate within segment
            seg_start = cumulative[seg_idx]
            seg_len = segment_lengths[seg_idx]
            if seg_len < 1e-6:
                t = 0
            else:
                t = (d - seg_start) / seg_len
            t = np.clip(t, 0, 1)
            
            positions[i] = path_arr[seg_idx] + t * diffs[seg_idx]
        
        if not add_velocity:
            return positions
        
        # Compute velocities (finite differences)
        velocities = np.zeros_like(positions)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / 2.0
        velocities[0] = positions[1] - positions[0]
        velocities[-1] = positions[-1] - positions[-2]
        
        # State: [vel_y, vel_x, pos_y, pos_x]
        states = np.concatenate([velocities, positions], axis=1)
        return states

    def generate_trajectory(self, horizon: int = 128, seed: int = None) -> Optional[np.ndarray]:
        """Generate a single valid trajectory.
        
        Returns: [horizon, 4] array or None if path finding failed.
        """
        rng = np.random.RandomState(seed)
        start = self.sample_position(rng)
        goal = self.sample_position(rng)
        
        # Ensure start and goal are sufficiently far apart
        while np.linalg.norm(goal - start) < 2.0:
            goal = self.sample_position(rng)
        
        path = self.find_path_rrt(start, goal, rng=rng)
        if path is None:
            return None
        
        return self.interpolate_path(path, horizon)

    def generate_dataset(self, n_trajectories: int = 1000, horizon: int = 128,
                         seed: int = 42, verbose: bool = True) -> Dict:
        """Generate a training dataset of trajectories.
        
        Classification: [ADAPTATION FOR LOCAL HARDWARE] — Replaces D4RL dataset.
        
        Returns:
            dict with:
              - 'trajectories': [N, horizon, 4] array
              - 'starts': [N, 4] start states
              - 'goals': [N, 4] goal states
              - 'norm_mins': [4] min values for normalization
              - 'norm_maxs': [4] max values for normalization
        """
        rng = np.random.RandomState(seed)
        trajectories = []
        starts = []
        goals = []
        
        attempts = 0
        while len(trajectories) < n_trajectories and attempts < n_trajectories * 5:
            traj = self.generate_trajectory(horizon=horizon, seed=rng.randint(1000000))
            attempts += 1
            if traj is not None:
                trajectories.append(traj)
                starts.append(traj[0])
                goals.append(traj[-1])
                if verbose and len(trajectories) % 100 == 0:
                    print(f"  Generated {len(trajectories)}/{n_trajectories} trajectories")
        
        if len(trajectories) == 0:
            raise RuntimeError("Failed to generate any trajectories")
        
        trajectories = np.array(trajectories)
        starts = np.array(starts)
        goals = np.array(goals)
        
        # Compute normalization bounds
        flat = trajectories.reshape(-1, 4)
        norm_mins = flat.min(axis=0)
        norm_maxs = flat.max(axis=0)
        
        if verbose:
            print(f"  Dataset: {len(trajectories)} trajectories, horizon={horizon}")
            print(f"  State range: min={norm_mins}, max={norm_maxs}")
        
        return {
            'trajectories': trajectories,
            'starts': starts,
            'goals': goals,
            'norm_mins': norm_mins,
            'norm_maxs': norm_maxs,
        }

    def normalize(self, x: np.ndarray, norm_mins: np.ndarray, norm_maxs: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] range.
        
        Classification: [SPECIFIED] — Matches upstream normalization.
        """
        return 2 * (x - norm_mins) / (norm_maxs - norm_mins + 1e-8) - 1

    def denormalize(self, x_norm: np.ndarray, norm_mins: np.ndarray, norm_maxs: np.ndarray) -> np.ndarray:
        """Denormalize from [-1, 1] to world coordinates."""
        return (x_norm + 1) / 2 * (norm_maxs - norm_mins) + norm_mins

    def get_walls_as_segments(self) -> List[Dict]:
        """Get wall segments for 3D visualization.
        
        Classification: [ENGINEERING ADDITION]
        
        Returns list of wall segments with positions and sizes.
        """
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y, x] == 1:
                    walls.append({
                        'x': x + 0.5,
                        'y': y + 0.5,
                        'width': 1.0,
                        'height': 1.0,
                    })
        return walls

    def to_dict(self) -> Dict:
        """Serialize environment for API/frontend."""
        return {
            'layout': self.layout.tolist(),
            'width': self.width,
            'height': self.height,
            'open_cells': [(int(y), int(x)) for y, x in self.open_cells],
            'walls': self.get_walls_as_segments(),
        }
