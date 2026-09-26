# SAFEGEN 3D — Research & Engineering Assumptions

**Reference Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)

---

## 1. Research & Scientific Assumptions

1. **Continuity & Differentiability of Safe Set Boundaries:**
   - The paper assumes the boundary of the safe set $\partial \mathcal{C}$ can be expressed as the zero level-set of a continuously differentiable function $b(\mathbf{x}) \in C^1$.
   - For polygonal or complex concave obstacles, safe sets are approximated via differentiable super-ellipsoids or unions of convex barriers.

2. **Kinematic Decoupling during Denoising:**
   - The reverse diffusion trajectory represents a spatio-temporal sequence $\boldsymbol{\tau} = (\mathbf{x}_0, \dots, \mathbf{x}_H)$.
   - While full dynamic constraints (joint torques, actuator limits) can be evaluated, the primary spatial collision avoidance operates at the positional coordinate slice $(x, y)$ of each waypoint.

3. **Time-Varying Relaxation Hypothesis (TVS):**
   - Requiring early noise iterates ($k \approx K$) to satisfy physical obstacle boundaries leads to mode collapse or severe bias towards obstacle centers.
   - The sigmoid relaxation $\sigma(k_{\mathrm{bias}} - k)$ properly delays the hard barrier constraint until the trajectory geometry has emerged ($k \le 10$ or $k \le 20$).

---

## 2. Local Hardware Adaptations `[ADAPTATION FOR LOCAL HARDWARE]`

1. **Operating System & MuJoCo Bypass:**
   - The upstream repository specifies Linux and MuJoCo 200 via `mujoco-py`. On Windows 11, `mujoco-py` compilation frequently fails due to MSVC header mismatches.
   - **Assumption:** A high-fidelity kinematic 2D maze simulator with identical coordinate bounds, wall collision geometry, and state normalization produces scientifically equivalent trajectory planning benchmarks.

2. **RTX 4050 6GB VRAM Optimization:**
   - Model hidden dimension is set to 32/64 channels (down from 256 in high-end cluster training) to guarantee sub-100ms inference and zero risk of Out-Of-Memory (OOM) errors.
   - Batch size for trajectory generation is set to 1 or 4 during interactive demo sessions.

---

## 3. Engineering Decisions `[ENGINEERING DECISION]`

1. **Analytical Dual Solver Default:**
   - For real-time web responsiveness ($< 100$ ms round-trip), the analytical closed-form KKT solver (`invariance_time_cf`) is prioritized over general interior-point QP solvers (`cvxpy`).
   - If more than 2 barriers become concurrently active, the system seamlessly falls back to CVXPY or sequential barrier projection.

2. **Closed-Loop Replanning Architecture:**
   - The paper focuses solely on offline, open-loop planning from fixed initial states.
   - In SAFEGEN 3D, we introduce an asynchronous event-driven Digital Twin state machine where dynamic obstacle insertions trigger automatic replanning from the current execution waypoint.
