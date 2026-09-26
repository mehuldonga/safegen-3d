# SAFEGEN 3D — Paper-to-Code Reference Guide

This document provides developers and reviewers with a direct mapping between the scientific claims and equations in the *SafeDiffuser* ICLR 2025 paper and the codebase.

For the exhaustive mathematical matrix with line numbers and test assertions, please refer to:
👉 **[research/PAPER_TO_CODE_MAP.md](../research/PAPER_TO_CODE_MAP.md)**

---

## Quick Reference Index

| Topic | Paper Location | SafeDiffuser Repo | SAFEGEN 3D Code |
|---|---|---|---|
| **DDPM Forward Schedule** | Section 3.1, Eq. 1 | `diffuser/models/diffusion.py:75` | [gaussian_diffusion.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/diffusion/gaussian_diffusion.py) |
| **Temporal UNet Backbone** | Section 3.1, Eq. 3 | `diffuser/models/temporal.py` | [temporal_unet.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/diffusion/temporal_unet.py) |
| **Super-Ellipsoid CBF** | Section 4.1, Eq. 8 | `diffuser/models/diffusion.py:830` | [barrier_functions.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/barrier_functions.py) |
| **Lie Spatial Gradients** | Section 4.1, Eq. 9 | `diffuser/models/diffusion.py:833` | [barrier_functions.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/barrier_functions.py) |
| **TVS Sigmoid Relaxation** | Section 4.3, Eq. 12 | `diffuser/models/diffusion.py:910` | [cbf_filter.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/cbf_filter.py) |
| **Analytical Dual KKT Solver** | Appendix C, Eq. 28 | `diffuser/models/diffusion.py:959` | [cbf_filter.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/cbf_filter.py) |
| **Batched QP Safety Filter** | Section 4.2, Eq. 14 | `diffuser/models/diffusion.py:880` | [cbf_filter.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/cbf_filter.py) |
| **Trajectory Validator** | Section 5.1 | `scripts/plan_maze2d.py:56` | [validators.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/safety/validators.py) |
| **Digital Twin World State** | Product Layer | None (Upstream is batch script) | [world_state.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/simulation/digital_twin/world_state.py) |
| **Interactive 3D Viewport** | Product Layer | None (Upstream outputs PNGs) | [App.jsx](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/frontend/src/App.jsx) |
| **Dynamic Obstacle Replanning**| Product Layer | None (Upstream is open-loop) | [planner_service.py](file:///c:/Users/kashy/OneDrive/Desktop/Research%20paper%20code/app/ml/planner/planner_service.py) |
