"""
Digital Twin — Stateful world model for the SAFEGEN 3D simulation.

Classification: [ENGINEERING ADDITION] — Product layer for interactive demonstration.

Manages:
  - Robot state (position, velocity, status)
  - Obstacles (static + dynamic)
  - Restricted zones
  - Trajectories (planned, executing, replanned)
  - Event system for state changes
  - Simulation stepping
"""

import time
import uuid
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from threading import Lock


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    ROBOT_STARTED = "ROBOT_STARTED"
    PLAN_GENERATED = "PLAN_GENERATED"
    PLAN_REJECTED = "PLAN_REJECTED"
    SAFETY_VIOLATION_DETECTED = "SAFETY_VIOLATION_DETECTED"
    OBSTACLE_ADDED = "OBSTACLE_ADDED"
    OBSTACLE_MOVED = "OBSTACLE_MOVED"
    OBSTACLE_REMOVED = "OBSTACLE_REMOVED"
    ENVIRONMENT_CHANGED = "ENVIRONMENT_CHANGED"
    REPLAN_REQUESTED = "REPLAN_REQUESTED"
    REPLAN_COMPLETED = "REPLAN_COMPLETED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    EXECUTION_PAUSED = "EXECUTION_PAUSED"
    EXECUTION_RESUMED = "EXECUTION_RESUMED"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    SIMULATION_RESET = "SIMULATION_RESET"


@dataclass
class Event:
    type: EventType
    timestamp: float
    data: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self):
        return {
            'type': self.type.value,
            'timestamp': self.timestamp,
            'data': self.data,
            'id': self.id,
        }


class EventBus:
    """Simple event bus for decoupled communication.
    
    Classification: [ENGINEERING ADDITION]
    """
    
    def __init__(self):
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._history: List[Event] = []
        self._lock = Lock()

    def subscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def emit(self, event: Event):
        with self._lock:
            self._history.append(event)
        for callback in self._listeners.get(event.type, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Event handler error: {e}")

    def get_history(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return [e.to_dict() for e in self._history[-limit:]]

    def clear_history(self):
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class SimulationStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RobotState:
    position: List[float] = field(default_factory=lambda: [1.5, 1.5])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    status: str = "idle"
    current_step: int = 0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Obstacle:
    id: str
    position: List[float]
    radius: float = 0.8
    shape: str = "ellipse"
    is_dynamic: bool = False
    color: str = "#ff4444"
    label: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RestrictedZone:
    id: str
    position: List[float]  # center
    size: List[float]       # [width, height]
    color: str = "#ff000040"
    label: str = "Restricted"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Trajectory:
    id: str
    points: List[List[float]]  # [[y, x], ...]
    type: str = "planned"      # planned, executing, replanned, rejected
    color: str = "#00ff88"
    is_safe: bool = True
    metrics: Dict = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'id': self.id,
            'points': self.points,
            'type': self.type,
            'color': self.color,
            'is_safe': self.is_safe,
            'metrics': self.metrics,
        }


# ---------------------------------------------------------------------------
# World State Manager
# ---------------------------------------------------------------------------

class WorldState:
    """Central state manager for the digital twin.
    
    Classification: [ENGINEERING ADDITION]
    
    Thread-safe world state with event-driven updates.
    """
    
    def __init__(self):
        self.robot = RobotState()
        self.obstacles: Dict[str, Obstacle] = {}
        self.restricted_zones: Dict[str, RestrictedZone] = {}
        self.trajectories: Dict[str, Trajectory] = {}
        self.goal: Optional[List[float]] = None
        self.start: Optional[List[float]] = None
        self.status = SimulationStatus.IDLE
        self.event_bus = EventBus()
        self.replan_count = 0
        self.execution_progress = 0.0
        self._lock = Lock()
        
        # Maze environment reference
        self.maze_layout = None
        self.maze_walls = []

    def reset(self, keep_static_obstacles: bool = False):
        """Reset the world to initial state."""
        with self._lock:
            self.robot = RobotState()
            if keep_static_obstacles:
                self.obstacles = {k: v for k, v in self.obstacles.items() if not v.is_dynamic}
            else:
                self.obstacles.clear()
            self.restricted_zones.clear()
            self.trajectories.clear()
            self.goal = None
            self.start = None
            self.status = SimulationStatus.IDLE
            self.replan_count = 0
            self.execution_progress = 0.0
        
        self.event_bus.emit(Event(
            type=EventType.SIMULATION_RESET,
            timestamp=time.time(),
        ))

    def set_mission(self, start: List[float], goal: List[float]):
        """Set the planning mission (start and goal positions)."""
        with self._lock:
            self.start = start
            self.goal = goal
            self.robot.position = start.copy()
            self.robot.status = "ready"

    def add_obstacle(self, position: List[float], radius: float = 0.8,
                     shape: str = "ellipse", is_dynamic: bool = False,
                     label: str = "") -> str:
        """Add an obstacle to the environment."""
        obs_id = f"obs_{len(self.obstacles) + 1}"
        obs = Obstacle(
            id=obs_id,
            position=position,
            radius=radius,
            shape=shape,
            is_dynamic=is_dynamic,
            label=label,
            color="#ff6644" if is_dynamic else "#ff4444",
        )
        with self._lock:
            self.obstacles[obs_id] = obs
        
        self.event_bus.emit(Event(
            type=EventType.OBSTACLE_ADDED,
            timestamp=time.time(),
            data={'obstacle': obs.to_dict()},
        ))
        return obs_id

    def remove_obstacle(self, obs_id: str):
        """Remove an obstacle."""
        with self._lock:
            if obs_id in self.obstacles:
                del self.obstacles[obs_id]
        self.event_bus.emit(Event(
            type=EventType.OBSTACLE_REMOVED,
            timestamp=time.time(),
            data={'obstacle_id': obs_id},
        ))

    def add_trajectory(self, points: List[List[float]], traj_type: str = "planned",
                       is_safe: bool = True, metrics: Dict = None) -> str:
        """Add a trajectory to the visualization."""
        traj_id = f"traj_{len(self.trajectories) + 1}"
        
        color_map = {
            'planned': '#00ff88',
            'executing': '#4488ff',
            'replanned': '#ffaa00',
            'rejected': '#ff4444',
            'vanilla': '#ff6666',
        }
        
        traj = Trajectory(
            id=traj_id,
            points=points,
            type=traj_type,
            color=color_map.get(traj_type, '#ffffff'),
            is_safe=is_safe,
            metrics=metrics or {},
        )
        
        with self._lock:
            self.trajectories[traj_id] = traj
        
        self.event_bus.emit(Event(
            type=EventType.PLAN_GENERATED,
            timestamp=time.time(),
            data={'trajectory': traj.to_dict()},
        ))
        return traj_id

    def update_robot_position(self, position: List[float], step: int):
        """Update robot position during execution."""
        with self._lock:
            self.robot.position = position
            self.robot.current_step = step
            self.robot.status = "moving"

    def check_collision_with_obstacles(self, position: List[float]) -> Optional[str]:
        """Check if a position collides with any dynamic obstacle.
        
        Returns the obstacle ID if collision detected, None otherwise.
        """
        for obs_id, obs in self.obstacles.items():
            dist = np.sqrt(
                (position[0] - obs.position[0]) ** 2 + 
                (position[1] - obs.position[1]) ** 2
            )
            if dist < obs.radius:
                return obs_id
        return None

    def check_trajectory_collision(self, trajectory_points: List[List[float]], 
                                    from_step: int = 0) -> Optional[int]:
        """Check if remaining trajectory collides with obstacles.
        
        Returns the step index of first collision, or None.
        """
        for i in range(from_step, len(trajectory_points)):
            if self.check_collision_with_obstacles(trajectory_points[i]):
                return i
        return None

    def to_dict(self) -> Dict:
        """Serialize complete world state for the frontend."""
        with self._lock:
            return {
                'robot': self.robot.to_dict(),
                'obstacles': {k: v.to_dict() for k, v in self.obstacles.items()},
                'restricted_zones': {k: v.to_dict() for k, v in self.restricted_zones.items()},
                'trajectories': {k: v.to_dict() for k, v in self.trajectories.items()},
                'goal': self.goal,
                'start': self.start,
                'status': self.status.value,
                'replan_count': self.replan_count,
                'execution_progress': self.execution_progress,
                'events': self.event_bus.get_history(20),
                'maze_walls': self.maze_walls,
            }
