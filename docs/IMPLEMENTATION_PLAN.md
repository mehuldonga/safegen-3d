# SAFEGEN 3D — Comprehensive 19-Phase Implementation Plan

---

## Overview
This roadmap governs the research reproduction, software engineering, and verification lifecycle of SAFEGEN 3D.

---

### PHASE 0: Environment Discovery
- **Objective:** Discover OS, Python, CUDA, GPU, Node.js, and hardware capacity.
- **Inputs:** Local system shell, hardware sensors.
- **Outputs:** Hardware profile (`configs/local_4050.yaml`), Python `.venv`, Node environment.
- **Acceptance Criteria:** Hardware detected as NVIDIA RTX 4050 Laptop GPU (6GB VRAM), Python 3.12, Node.js v18+.
- **Recovery Strategy:** Fallback to CPU execution if CUDA runtime is unavailable.

### PHASE 1: Paper Analysis
- **Objective:** Extract equations, problem definition, CBF derivations, and theoretical bounds from *SafeDiffuser* ICLR 2025.
- **Inputs:** Official ICLR PDF, project website, conference proceedings.
- **Outputs:** `research/PAPER_ANALYSIS.md`, `research/MATHEMATICAL_SPECIFICATION.md`.
- **Acceptance Criteria:** Every equation classified as `[SPECIFIED]`, `[ADAPTATION]`, or `[ENGINEERING DECISION]`.

### PHASE 2: Upstream Repository Inspection
- **Objective:** Clone and inspect upstream repository line-by-line without altering reference code.
- **Inputs:** `https://github.com/Weixy21/SafeDiffuser`.
- **Outputs:** Preserved clone in `upstream/SafeDiffuser/`, `research/REPRODUCTION_NOTES.md`.
- **Acceptance Criteria:** Direct code mapping identified for `diffuser/models/diffusion.py` lines 830–1095.

### PHASE 3: Minimal Research Reproduction
- **Objective:** Build self-contained 2D kinematic simulation matching D4RL `maze2d-large-v1` coordinates without MuJoCo.
- **Outputs:** `app/simulation/maze/maze_env.py`.
- **Tests:** Coordinate boundary tests, wall collision queries.
- **Acceptance Criteria:** 12x9 maze layout with identical wall segments.

### PHASE 4: Vanilla Diffusion Baseline
- **Objective:** Implement 1D Temporal UNet, cosine/linear noise schedules, and forward/reverse sampling.
- **Outputs:** `app/ml/diffusion/temporal_unet.py`, `app/ml/diffusion/gaussian_diffusion.py`.
- **Tests:** `tests/ml/test_diffusion.py`.
- **Acceptance Criteria:** Reverse sampling generates valid trajectory arrays `[B, H, D]`.

### PHASE 5: SafeDiffuser CBF Integration
- **Objective:** Implement Control Barrier Functions, Lie derivatives, TVS relaxation, and KKT dual solver.
- **Outputs:** `app/ml/safety/barrier_functions.py`, `app/ml/safety/cbf_filter.py`.
- **Tests:** `tests/ml/test_safety.py`.
- **Acceptance Criteria:** Analytical KKT solver accurately repels nominal step away from obstacles.

### PHASE 6: Safety Validator
- **Objective:** Implement post-generation safety validator computing minimum margin and collision count.
- **Outputs:** `app/ml/safety/validators.py`.
- **Tests:** `test_trajectory_validator_metrics`.
- **Acceptance Criteria:** Flags collisions ($b < 0$) and certifies safe trajectories ($b \ge 0$).

### PHASE 7: Simulation & Digital Twin Engine
- **Objective:** Create thread-safe world state manager and event bus for robot, obstacles, and paths.
- **Outputs:** `app/simulation/digital_twin/world_state.py`.
- **Tests:** `tests/integration/test_replanning.py`.
- **Acceptance Criteria:** Emits events (`OBSTACLE_ADDED`, `REPLAN_REQUESTED`) with timestamped JSON history.

### PHASE 8: 3D Visualization Scaffolding
- **Objective:** Build interactive React Three Fiber 3D scene with maze walls, floor grid, beacons, and robot.
- **Outputs:** `app/frontend/src/App.jsx`, `App.css`.
- **Tests:** `npm run build`.
- **Acceptance Criteria:** Production bundle compiles cleanly in $< 5$s.

### PHASE 9: FastAPI Backend
- **Objective:** Implement REST API and WebSocket broadcast server for real-time digital twin synchronization.
- **Outputs:** `app/backend/main.py`.
- **Tests:** `tests/api/test_endpoints.py`.
- **Acceptance Criteria:** All endpoints return 200 on valid input and handle errors safely.

### PHASE 10: Frontend Dashboard & Controls
- **Objective:** Build mission control panel, mode toggles, real-time metrics bar, and status pills.
- **Outputs:** `app/frontend/src/App.jsx`, `App.css`.
- **Acceptance Criteria:** Premium dark theme aesthetic with responsive UI panels.

### PHASE 11: Dynamic Obstacles
- **Objective:** Implement real-time in-flight obstacle injection capability.
- **Outputs:** `/simulation/obstacle` endpoint, frontend "INJECT OBSTACLE" button.
- **Acceptance Criteria:** Obstacle appears instantly in 3D canvas and digital twin state.

### PHASE 12: Automatic Replanning
- **Objective:** Trigger replanning when active trajectory intersects newly injected dynamic hazard.
- **Outputs:** `app/ml/planner/planner_service.py`, `app/simulation/digital_twin/world_state.py`.
- **Tests:** `test_hero_scenario_event_flow`.
- **Acceptance Criteria:** Robot halts, SafeDiffuser replans detour path, robot resumes to goal.

### PHASE 13: Metrics & Comparative Benchmarking
- **Objective:** Collect and compare real planning times, safety margins, and violation counts.
- **Outputs:** `GET /metrics`, comparative overlay in 3D canvas.
- **Acceptance Criteria:** Vanilla diffuser exhibits safety violations while SafeDiffuser achieves 0 violations.

### PHASE 14: Verification Agent
- **Objective:** Build autonomous test runner evaluating requirements REQ-001 through REQ-012.
- **Outputs:** `verification/verification_agent.py`, `scripts/run_verification.ps1`.
- **Acceptance Criteria:** Produces `results.json` and `report.md`.

### PHASE 15: Automated Repair Loop
- **Objective:** Automatically diagnose, patch, and re-test any failing requirement.
- **Outputs:** `verification/failures.md`.
- **Acceptance Criteria:** Zero unresolved test failures.

### PHASE 16: End-to-End Integration
- **Objective:** Connect frontend, backend, ML core, and digital twin on localhost.
- **Outputs:** `scripts/start_dev.ps1`, `scripts/stop_dev.ps1`.
- **Acceptance Criteria:** Single-command startup on Windows.

### PHASE 17: Research Validation
- **Objective:** Final audit comparing codebase to ICLR 2025 paper equations and upstream implementation.
- **Outputs:** `docs/FINAL_RESEARCH_REVIEW.md`.
- **Acceptance Criteria:** 100% scientific fidelity on ported equations.

### PHASE 18: Documentation
- **Objective:** Author full suite of research specifications, model cards, setup guides, and demo scripts.
- **Outputs:** 15 documentation files in `docs/` and 7 in `research/`.
- **Acceptance Criteria:** All documentation verified on disk.

### PHASE 19: Final Demo
- **Objective:** Execute live 3-5 minute demonstration of Hero Scenario on localhost.
- **Outputs:** `docs/DEMO_SCRIPT.md`, live demo at `http://localhost:5173`.
- **Acceptance Criteria:** Seamless visual and technical execution.
