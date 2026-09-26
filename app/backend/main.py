"""
SAFEGEN 3D — FastAPI Backend

Classification: [ENGINEERING ADDITION] — Product API layer.

Provides REST endpoints and WebSocket for:
  - Planning (generate/validate/replan trajectories)
  - Simulation (start/stop/inject obstacles)
  - Metrics and experiment tracking
  - Real-time state streaming via WebSocket
"""

import sys
import os
import time
import json
import asyncio
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.simulation.digital_twin.world_state import WorldState, SimulationStatus, EventType, Event
from app.simulation.maze.maze_env import Maze2DEnvironment, MAZE2D_LARGE_LAYOUT

logger = logging.getLogger("safegen3d")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

world = WorldState()
maze_env = Maze2DEnvironment(MAZE2D_LARGE_LAYOUT)
planner = None  # Lazy-loaded
connected_clients: List[WebSocket] = []
simulation_task = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    gpu_available: bool = False
    planner_ready: bool = False
    simulation_status: str = "idle"

class MissionRequest(BaseModel):
    start: List[float] = Field(..., description="[y, x] start position")
    goal: List[float] = Field(..., description="[y, x] goal position")

class PlanRequest(BaseModel):
    planner_mode: str = "safe_diffuser"
    performance_mode: str = "balanced"
    seed: Optional[int] = None
    batch_size: int = 4

class ObstacleRequest(BaseModel):
    position: List[float] = Field(..., description="[y, x] position")
    radius: float = 0.8
    shape: str = "ellipse"
    label: str = ""

class SimulationControlRequest(BaseModel):
    action: str = "start"  # start, pause, resume, stop

class MetricsResponse(BaseModel):
    planning_time_ms: float = 0
    trajectory_length: float = 0
    safety_violations: int = 0
    min_safety_margin: float = 0
    collision_checks: int = 0
    replan_count: int = 0
    execution_progress: float = 0
    goal_distance: Optional[float] = None


# ---------------------------------------------------------------------------
# Lifespan (initialization)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the planner on startup."""
    global planner
    logger.info("SAFEGEN 3D Backend starting...")
    
    # Initialize maze environment
    world.maze_layout = maze_env.layout.tolist()
    world.maze_walls = maze_env.get_walls_as_segments()
    
    # Initialize default mission and static research barriers
    world.set_mission([1.5, 1.5], [7.5, 10.5])
    world.add_obstacle(position=[5.0, 5.8], radius=1.0, shape='ellipse', label="Research Barrier 1")
    world.add_obstacle(position=[2.0, 5.3], radius=1.0, shape='superellipse', label="Research Barrier 2")
    
    # Lazy-init planner (will be initialized on first planning request)
    logger.info("Maze environment ready with default mission & barriers. Planner will initialize on first use.")
    
    yield
    
    logger.info("SAFEGEN 3D Backend shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAFEGEN 3D API",
    description="SafeDiffuser-based safe generative planning with 3D digital twin",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: ensure planner is initialized
# ---------------------------------------------------------------------------

def ensure_planner():
    """Lazy-initialize the planner service."""
    global planner
    if planner is not None:
        return
    
    try:
        from app.ml.planner.planner_service import PlannerService
        planner = PlannerService(device='auto')
        planner.initialize(horizon=64)  # [ADAPTATION FOR LOCAL HARDWARE] reduced horizon
        logger.info(f"Planner initialized on {planner.device}")
    except Exception as e:
        logger.error(f"Failed to initialize planner: {e}")
        raise HTTPException(status_code=503, detail=f"Planner initialization failed: {str(e)}")


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------

async def broadcast_state():
    """Send current world state to all connected WebSocket clients."""
    if not connected_clients:
        return
    state_json = json.dumps(world.to_dict())
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(state_json)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        planner_ready=planner is not None and planner.is_initialized,
        simulation_status=world.status.value,
    )


@app.get("/environment")
async def get_environment():
    """Get the maze environment layout."""
    return maze_env.to_dict()


@app.post("/planning/create")
async def create_mission(request: MissionRequest):
    """Create a new planning mission with start and goal."""
    world.reset(keep_static_obstacles=True)
    world.set_mission(request.start, request.goal)
    await broadcast_state()
    return {"status": "ok", "start": request.start, "goal": request.goal}


@app.post("/planning/generate")
async def generate_plan(request: PlanRequest):
    """Generate a trajectory using the diffusion planner."""
    ensure_planner()
    
    if world.start is None or world.goal is None:
        raise HTTPException(400, "No mission set. Call /planning/create first.")
    
    world.status = SimulationStatus.PLANNING
    await broadcast_state()
    
    try:
        from app.ml.planner.planner_service import PlanningRequest, PlannerMode, PerformanceMode
        
        planning_req = PlanningRequest(
            start_pos=world.start,
            goal_pos=world.goal,
            planner_mode=PlannerMode(request.planner_mode),
            performance_mode=PerformanceMode(request.performance_mode),
            seed=request.seed,
            batch_size=request.batch_size,
        )
        
        result = planner.plan(planning_req)
        
        # Extract position-only trajectory for visualization
        traj_points = [[p[2], p[3]] for p in result.trajectory]  # [pos_y, pos_x]
        
        # Add trajectory to world
        traj_type = "planned" if request.planner_mode == "safe_diffuser" else "vanilla"
        world.add_trajectory(
            points=traj_points,
            traj_type=traj_type,
            is_safe=result.is_safe,
            metrics=result.metrics,
        )
        
        # Also add vanilla trajectory for comparison if using safe mode
        if request.planner_mode == "safe_diffuser" and result.all_trajectories:
            # Generate a vanilla trajectory for comparison
            try:
                vanilla_req = PlanningRequest(
                    start_pos=world.start,
                    goal_pos=world.goal,
                    planner_mode=PlannerMode.VANILLA,
                    performance_mode=PerformanceMode(request.performance_mode),
                    seed=request.seed,
                    batch_size=2,
                )
                vanilla_result = planner.plan(vanilla_req)
                vanilla_points = [[p[2], p[3]] for p in vanilla_result.trajectory]
                world.add_trajectory(
                    points=vanilla_points,
                    traj_type="vanilla",
                    is_safe=vanilla_result.is_safe,
                    metrics=vanilla_result.metrics,
                )
            except Exception as e:
                logger.warning(f"Failed to generate vanilla comparison: {e}")
        
        world.status = SimulationStatus.IDLE
        await broadcast_state()
        
        return result.to_dict()
        
    except Exception as e:
        world.status = SimulationStatus.FAILED
        await broadcast_state()
        logger.error(f"Planning failed: {e}", exc_info=True)
        raise HTTPException(500, f"Planning failed: {str(e)}")


@app.post("/planning/validate")
async def validate_plan():
    """Validate the current trajectory against safety constraints."""
    if not world.trajectories:
        raise HTTPException(400, "No trajectory to validate.")
    
    # Get the latest planned trajectory
    latest = list(world.trajectories.values())[-1]
    return {
        "is_safe": latest.is_safe,
        "metrics": latest.metrics,
        "trajectory_type": latest.type,
    }


@app.post("/planning/replan")
async def replan():
    """Trigger replanning (e.g., after obstacle injection)."""
    ensure_planner()
    
    if world.start is None or world.goal is None:
        raise HTTPException(400, "No mission set.")
    
    world.status = SimulationStatus.REPLANNING
    world.replan_count += 1
    
    world.event_bus.emit(Event(
        type=EventType.REPLAN_REQUESTED,
        timestamp=time.time(),
        data={'reason': 'environment_changed', 'replan_count': world.replan_count}
    ))
    
    await broadcast_state()
    
    try:
        from app.ml.planner.planner_service import PlanningRequest, PlannerMode, PerformanceMode
        
        # Use current robot position as new start
        current_pos = world.robot.position
        
        planning_req = PlanningRequest(
            start_pos=current_pos,
            goal_pos=world.goal,
            planner_mode=PlannerMode.SAFE_DIFFUSER,
            performance_mode=PerformanceMode.FAST_DEMO,  # Fast for replanning
            batch_size=2,
        )
        
        # Add dynamic obstacles to the planner
        for obs in world.obstacles.values():
            if obs.is_dynamic:
                planner.add_obstacle_barrier(
                    center_y=obs.position[0],
                    center_x=obs.position[1],
                    radius=obs.radius,
                    shape=obs.shape,
                )
        
        result = planner.plan(planning_req)
        
        traj_points = [[p[2], p[3]] for p in result.trajectory]
        world.add_trajectory(
            points=traj_points,
            traj_type="replanned",
            is_safe=result.is_safe,
            metrics=result.metrics,
        )
        
        world.event_bus.emit(Event(
            type=EventType.REPLAN_COMPLETED,
            timestamp=time.time(),
            data={'is_safe': result.is_safe},
        ))
        
        world.status = SimulationStatus.IDLE
        await broadcast_state()
        
        return result.to_dict()
        
    except Exception as e:
        world.status = SimulationStatus.FAILED
        await broadcast_state()
        raise HTTPException(500, f"Replanning failed: {str(e)}")


@app.post("/simulation/reset")
async def reset_simulation():
    """Reset the simulation to initial state."""
    global simulation_task
    if simulation_task is not None:
        simulation_task.cancel()
        simulation_task = None
    world.reset(keep_static_obstacles=True)
    if not world.obstacles:
        world.add_obstacle(position=[5.0, 5.8], radius=1.0, shape='ellipse', label="Research Barrier 1")
        world.add_obstacle(position=[2.0, 5.3], radius=1.0, shape='superellipse', label="Research Barrier 2")
    world.set_mission([1.5, 1.5], [7.5, 10.5])
    await broadcast_state()
    return {"status": "ok"}


@app.post("/simulation/obstacle")
async def add_obstacle(request: ObstacleRequest):
    """Add a dynamic obstacle (for the 'inject obstacle' feature)."""
    obs_id = world.add_obstacle(
        position=request.position,
        radius=request.radius,
        shape=request.shape,
        is_dynamic=True,
        label=request.label or "Dynamic Obstacle",
    )
    
    # Check if current trajectory is affected
    needs_replan = False
    for traj in list(world.trajectories.values()):
        if traj.type in ('planned', 'replanned', 'executing'):
            collision_step = world.check_trajectory_collision(traj.points)
            if collision_step is not None:
                needs_replan = True
                world.event_bus.emit(Event(
                    type=EventType.ENVIRONMENT_CHANGED,
                    timestamp=time.time(),
                    data={
                        'obstacle_id': obs_id,
                        'collision_at_step': collision_step,
                        'needs_replan': True,
                    }
                ))
    
    await broadcast_state()

    # Automatically trigger replanning if trajectory is compromised
    if needs_replan:
        try:
            logger.info("Dynamic obstacle intersects trajectory! Triggering auto-replan...")
            await replan()
        except Exception as e:
            logger.warning(f"Auto-replanning failed: {e}")

    return {"status": "ok", "obstacle_id": obs_id, "message": "Obstacle injected", "replanned": needs_replan}


@app.post("/simulation/start")
async def start_simulation():
    """Start executing the planned trajectory."""
    global simulation_task
    
    # Find any planned, replanned, or vanilla trajectory to execute
    trajs = [t for t in world.trajectories.values() 
             if t.type in ('planned', 'replanned', 'vanilla')]
    
    if not trajs:
        raise HTTPException(400, "No trajectory to execute. Generate a plan first.")
    
    # Prefer safe trajectory if available, otherwise take the latest generated
    safe_trajs = [t for t in trajs if t.is_safe]
    latest_traj = safe_trajs[-1] if safe_trajs else trajs[-1]
    
    world.status = SimulationStatus.EXECUTING
    world.robot.status = "moving"
    latest_traj.type = "executing"
    
    world.event_bus.emit(Event(
        type=EventType.EXECUTION_STARTED,
        timestamp=time.time(),
    ))
    
    await broadcast_state()
    
    # Start async execution
    simulation_task = asyncio.create_task(
        execute_trajectory(latest_traj)
    )
    
    return {"status": "executing", "trajectory_id": latest_traj.id}


async def execute_trajectory(trajectory):
    """Asynchronously execute a trajectory, updating world state."""
    points = trajectory.points
    total_steps = len(points)
    
    try:
        for step in range(total_steps):
            if world.status == SimulationStatus.PAUSED:
                # Wait while paused
                while world.status == SimulationStatus.PAUSED:
                    await asyncio.sleep(0.1)
            
            if world.status not in (SimulationStatus.EXECUTING,):
                break
            
            pos = points[step]
            world.update_robot_position(pos, step)
            world.execution_progress = (step + 1) / total_steps
            
            # Check for collision with dynamic obstacles
            collision_id = world.check_collision_with_obstacles(pos)
            if collision_id is not None:
                if trajectory.metrics is None:
                    trajectory.metrics = {}
                trajectory.metrics['safety_violations'] = trajectory.metrics.get('safety_violations', 0) + 1
                trajectory.is_safe = False
                world.event_bus.emit(Event(
                    type=EventType.SAFETY_VIOLATION_DETECTED,
                    timestamp=time.time(),
                    data={'obstacle_id': collision_id, 'step': step, 'position': pos}
                ))
            
            await broadcast_state()
            await asyncio.sleep(0.05)  # ~20 FPS simulation rate
        
        if world.status == SimulationStatus.EXECUTING:
            world.status = SimulationStatus.COMPLETED
            world.robot.status = "arrived"
            world.event_bus.emit(Event(
                type=EventType.EXECUTION_COMPLETED,
                timestamp=time.time(),
            ))
            await broadcast_state()
            
    except asyncio.CancelledError:
        world.status = SimulationStatus.PAUSED
        await broadcast_state()
    except Exception as e:
        world.status = SimulationStatus.FAILED
        world.event_bus.emit(Event(
            type=EventType.EXECUTION_FAILED,
            timestamp=time.time(),
            data={'error': str(e)},
        ))
        await broadcast_state()


@app.post("/simulation/pause")
async def pause_simulation():
    """Pause trajectory execution."""
    world.status = SimulationStatus.PAUSED
    world.event_bus.emit(Event(
        type=EventType.EXECUTION_PAUSED,
        timestamp=time.time(),
    ))
    await broadcast_state()
    return {"status": "paused"}


@app.get("/simulation/state")
async def get_simulation_state():
    """Get the current simulation state."""
    return world.to_dict()


@app.get("/planning/status")
async def get_planning_status():
    """Get current planning/simulation status."""
    return {
        "status": world.status.value,
        "planner_ready": planner is not None and planner.is_initialized,
        "replan_count": world.replan_count,
        "execution_progress": world.execution_progress,
        "trajectory_count": len(world.trajectories),
        "obstacle_count": len(world.obstacles),
    }


@app.get("/metrics")
async def get_metrics():
    """Get current metrics."""
    latest_metrics = {}
    for traj in world.trajectories.values():
        if traj.metrics:
            latest_metrics = traj.metrics
    
    return {
        "planning_time_ms": latest_metrics.get('planning_time_ms', 0),
        "trajectory_length": latest_metrics.get('trajectory_length', 0),
        "safety_violations": latest_metrics.get('safety_violations', 0),
        "min_safety_margin": latest_metrics.get('min_safety_margin', 0),
        "replan_count": world.replan_count,
        "execution_progress": world.execution_progress,
        "goal_distance": latest_metrics.get('goal_distance', None),
    }


@app.get("/experiments")
async def get_experiments():
    """Get experiment history."""
    experiments = []
    for traj_id, traj in world.trajectories.items():
        experiments.append({
            'id': traj_id,
            'type': traj.type,
            'is_safe': traj.is_safe,
            'metrics': traj.metrics,
        })
    return {"experiments": experiments}


@app.get("/events")
async def get_events():
    """Get event history."""
    return {"events": world.event_bus.get_history(50)}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time simulation state updates."""
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected ({len(connected_clients)} total)")
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps(world.to_dict()))
        
        while True:
            # Listen for client messages (e.g., ping)
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected ({len(connected_clients)} total)")
    except Exception as e:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.error(f"WebSocket error: {e}")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
