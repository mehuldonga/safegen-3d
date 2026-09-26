# SAFEGEN 3D — Research Reproduction Notes

**Reference Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)

---

## 1. Upstream Repository Inspection & Forensics

The official repository `https://github.com/Weixy21/SafeDiffuser` was cloned into `upstream/SafeDiffuser/` and subjected to line-by-line inspection.

### Key Architectural Findings:
1. **Model Backbone (`diffuser/models/temporal.py`):**
   - Implements `TemporalUnet` consisting of 1D temporal convolutions (`Conv1d`), sinusoidal timestep embeddings, residual blocks with GroupNorm and Mish activation, and down/up sampling blocks.
   - Operates on tensor dimensions `[batch_size, horizon, transition_dim]`.
2. **Diffusion Engine (`diffuser/models/diffusion.py`):**
   - DDPM Gaussian diffusion model parameterized with $\beta$ variance schedule.
   - At line 1030 (`p_sample`), the nominal model mean and variance are computed.
   - Lines 1040-1096 contain commented conditional branches switching between unconstrained sampling, classifier guidance (`GD`), truncation shielding, and SafeDiffuser invariance methods (`invariance`, `invariance_cf`, `invariance_relax`, `invariance_time`).
   - Line 1072 reveals an important empirical insight: **Applying CBF safety filtering only during the final 10 steps ($k \le 10$) yields the best trade-off between generative trajectory diversity and strict safety compliance.**
3. **Safety Optimization (`diffuser/models/diffusion.py:830-1030`):**
   - Obstacles are parameterized as super-ellipsoids:
     $$b_m(x, y) = ((y - y_c)/r_y)^p + ((x - x_c)/r_x)^p - 1 - \text{margin}$$
   - Spatial gradients (`Lgbu1`, `Lgbu2`) and time-varying derivative (`Lfb`) are computed explicitly.
   - A batched QP is formed using `qpth.qp.QPFunction` or solved via an analytical closed-form formula (`invariance_time_cf` at line 959).
4. **Environment Constraints (`scripts/plan_maze2d.py`):**
   - Upstream scripts rely heavily on MuJoCo 200 binaries (`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin`) and D4RL (`gym.make('maze2d-large-v1')`).
   - These libraries are Linux-native and incompatible with Windows environments without specialized C++ compilers and deprecated license wrappers.

---

## 2. Adaptation Strategy for Local Execution

In strict adherence to the directive's non-negotiable rules:
1. **Algorithm Fidelity:** The mathematical formulations of the diffusion process, score model, CBF safety filters, time-varying relaxations, and Lie derivatives are preserved verbatim.
2. **Local Simulation Engine:** We engineered a self-contained 2D kinematic environment (`app/simulation/maze/maze_env.py`) that replicates the exact coordinate scale, wall topology, and navigation dynamics of D4RL `maze2d-large-v1` without requiring MuJoCo binaries.
3. **Solver Flexibility:** We support both the analytical closed-form KKT solver and a Python-native `cvxpy`/`scipy` quadratic program fallback, removing the fragile `qpth` C++ CUDA dependency while retaining exact QP optimality.
4. **Interactive Digital Twin:** While upstream produces static PNG plots after offline planning, SAFEGEN 3D wraps the planner in an interactive, event-driven Digital Twin with real-time 3D Three.js rendering and dynamic obstacle injection.

---

## 3. Reproduction Audit Table

| Component | Paper Reference | Upstream Implementation | SAFEGEN 3D Status |
|---|---|---|---|
| Temporal UNet | Sec. 3.1 | `diffuser/models/temporal.py` | Fully ported in PyTorch |
| Gaussian Diffusion | Sec. 3.1 | `diffuser/models/diffusion.py` | Fully ported in PyTorch |
| Super-Ellipsoid CBF | Sec. 5.1 | `diffuser/models/diffusion.py:830` | Implemented in `barrier_functions.py` |
| TVS Sigmoid Relaxation | Sec. 4.3 | `diffuser/models/diffusion.py:910` | Implemented in `cbf_filter.py` |
| Analytical Dual KKT Solver | App. C | `diffuser/models/diffusion.py:959` | Implemented in `cbf_filter.py` |
| Maze2D Environment | Sec. 5.1 | D4RL `maze2d-large-v1` | Self-contained equivalent |
| Real-time Dynamic Replanning | Beyond paper | None (open-loop in paper) | Implemented in `world_state.py` & backend |
| 3D Digital Twin | Beyond paper | None | Implemented in React + Three.js |
