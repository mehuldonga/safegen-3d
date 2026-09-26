# SAFEGEN 3D — Paper-to-Code Traceability Map

This document establishes bidirectional traceability between every research claim, equation, and algorithmic step in *SafeDiffuser* (ICLR 2025) and our implementation.

---

## 1. Traceability Matrix

| Research Concept | Paper Section | Upstream SafeDiffuser | SAFEGEN 3D Module | Verification Test | Classification |
|---|---|---|---|---|---|
| **Trajectory Representation** $\boldsymbol{\tau} \in \mathbb{R}^{H \times D}$ | Sec. 3.1 | `diffuser/models/diffusion.py:270` | `app/ml/diffusion/gaussian_diffusion.py` | `tests/ml/test_diffusion.py::test_trajectory_shape` | `[SPECIFIED]` |
| **Forward Noise Schedule** $\beta_k, \alpha_k, \bar{\alpha}_k$ | Sec. 3.1 | `diffuser/models/diffusion.py:75-110` | `app/ml/diffusion/gaussian_diffusion.py` | `tests/ml/test_diffusion.py::test_noise_schedule` | `[SPECIFIED]` |
| **Temporal 1D Conv UNet** | Sec. 3.1 | `diffuser/models/temporal.py` | `app/ml/diffusion/temporal_unet.py` | `tests/ml/test_diffusion.py::test_unet_forward` | `[SPECIFIED]` |
| **Reverse Denoising Step** | Sec. 3.1 | `diffuser/models/diffusion.py:1030` | `app/ml/diffusion/gaussian_diffusion.py:p_sample` | `tests/ml/test_diffusion.py::test_p_sample` | `[SPECIFIED]` |
| **Start/Goal Conditioning** | Sec. 3.2 | `diffuser/models/helpers.py:apply_conditioning` | `app/ml/diffusion/gaussian_diffusion.py:apply_conditioning` | `tests/ml/test_diffusion.py::test_conditioning` | `[SPECIFIED]` |
| **Super-Ellipsoid CBF** $b_m(\mathbf{x})$ | Sec. 4.1 | `diffuser/models/diffusion.py:830-855` | `app/ml/safety/barrier_functions.py:SuperEllipsoidBarrier` | `tests/ml/test_safety.py::test_barrier_evaluation` | `[SPECIFIED]` |
| **Barrier Spatial Gradients** $\nabla_{\mathbf{x}} b_m$ | Sec. 4.1 | `diffuser/models/diffusion.py:833-834` | `app/ml/safety/barrier_functions.py:gradient` | `tests/ml/test_safety.py::test_barrier_gradient` | `[SPECIFIED]` |
| **TVS Time-Varying Relaxation** | Sec. 4.3 | `diffuser/models/diffusion.py:910-915` | `app/ml/safety/cbf_filter.py:compute_tvs_barrier` | `tests/ml/test_safety.py::test_tvs_relaxation` | `[SPECIFIED]` |
| **CBF Inequality Constraint** $\mathbf{G} \mathbf{u} \le \mathbf{h}$ | Sec. 4.2 | `diffuser/models/diffusion.py:856-868` | `app/ml/safety/cbf_filter.py:build_cbf_constraints` | `tests/ml/test_safety.py::test_cbf_constraints` | `[SPECIFIED]` |
| **Quadratic Program Filter** | Sec. 4.2 | `diffuser/models/diffusion.py:880-890` | `app/ml/safety/cbf_filter.py:solve_qp_filter` | `tests/ml/test_safety.py::test_qp_safety_filter` | `[SPECIFIED]` |
| **Analytical Dual KKT Solver** | App. C | `diffuser/models/diffusion.py:959-1025` | `app/ml/safety/cbf_filter.py:solve_closed_form_tvs` | `tests/ml/test_safety.py::test_analytical_kkt` | `[SPECIFIED]` |
| **Post-Gen Safety Validator** | Sec. 5.1 | `scripts/plan_maze2d.py:56-65` | `app/ml/safety/validators.py:TrajectoryValidator` | `tests/ml/test_safety.py::test_trajectory_validator` | `[ENGINEERING ADDITION]` |
| **Kinematic Maze2D Simulator** | Sec. 5.1 | D4RL `maze2d-large-v1` | `app/simulation/maze/maze_env.py` | `tests/simulation/test_maze.py` | `[ADAPTATION FOR LOCAL HARDWARE]` |
| **Dynamic Obstacle Injection** | Demo Req. | None (Not in paper) | `app/simulation/digital_twin/world_state.py` | `tests/integration/test_replanning.py` | `[ENGINEERING ADDITION]` |
| **Closed-Loop Replanning Loop** | Demo Req. | None (Paper is open-loop) | `app/ml/planner/planner_service.py` | `tests/integration/test_replanning.py` | `[ENGINEERING ADDITION]` |
| **FastAPI REST + WebSocket** | Product Req. | None | `app/backend/main.py` | `tests/api/test_endpoints.py` | `[ENGINEERING ADDITION]` |
| **3D Three.js Digital Twin** | Visual Req. | None | `app/frontend/src/App.jsx` | UI Browser Test | `[ENGINEERING ADDITION]` |

---

## 2. Classification Legend

- **`[SPECIFIED]`**: Directly derived from theoretical claims, equations, and code in the official SafeDiffuser ICLR 2025 publication.
- **`[ADAPTATION FOR LOCAL HARDWARE]`**: Modified specifically to ensure reliable execution on Windows and local consumer GPU (RTX 4050) without sacrificing mathematical fidelity.
- **`[ENGINEERING ADDITION]`**: Modern software architecture, interactive digital twin, and full-stack product interfaces built around the research core.
