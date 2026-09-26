# SAFEGEN 3D — Experiment Specification & Benchmark Protocol

**Reference Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)

---

## 1. Experimental Objectives

1. **Safety Verification:** Demonstrate that when a hazardous restricted zone or dynamic obstacle lies directly between the start state $\mathbf{s}_0$ and goal $\mathbf{s}_H$:
   - The unconstrained **Vanilla Diffuser** frequently cuts through the restricted area (safety violations $> 0$, minimum margin $< 0$).
   - The **SafeDiffuser CBF Filter** proactively bends the trajectory manifold around the barrier, guaranteeing zero safety violations ($b_m(\mathbf{x}) \ge 0$).
2. **Dynamic Replanning Validation:** Demonstrate that upon real-time obstacle injection directly intersecting the active robot trajectory, the system pauses, detects the collision hazard, triggers SafeDiffuser replanning from the current state, and reaches the goal safely.
3. **Computational Benchmarking:** Measure planning latency, number of QP iterations, and runtime on local RTX 4050 GPU / CPU across different performance modes.

---

## 2. Experimental Scenarios

### Scenario A: The Hero Demonstration (Restricted Zone Corridor)
- **Start State:** $\mathbf{s}_0 = (1.0, 1.0)$
- **Goal State:** $\mathbf{s}_H = (7.0, 7.0)$
- **Restricted Zone 1:** Center $(4.0, 4.0)$, Semi-axes $(1.2, 1.2)$, Order $p=2$, Margin buffer $\delta = 0.05$.
- **Restricted Zone 2:** Center $(4.5, 2.5)$, Semi-axes $(1.0, 0.8)$, Order $p=4$, Margin buffer $\delta = 0.05$.
- **Expected Outcome:**
  - Vanilla Diffuser generates a direct linear path connecting start and goal, cutting directly through Restricted Zone 1.
  - SafeDiffuser repels intermediate trajectory points outside both restricted zones, synthesizing a smooth arc around the perimeter.

### Scenario B: Dynamic In-Flight Obstacle Injection
- **Initial Mission:** Robot begins traversing SafeDiffuser plan from Scenario A.
- **Trigger Event:** At $t = 3.5\text{s}$ (approx. 40% execution progress), user clicks **INJECT OBSTACLE**.
- **Injected Obstacle:** Dynamic hazard spawns at coordinates $(4.0, 6.2)$, radius $0.9$.
- **System Behavior:**
  1. Sensor / Digital Twin detects intersection: distance to obstacle boundary $< 0.1$.
  2. Event `SAFETY_VIOLATION_DETECTED` and `REPLAN_REQUESTED` fired.
  3. Robot velocity arrested.
  4. SafeDiffuser replanner receives current robot position $\mathbf{s}_{\text{robot}}$ as new start condition.
  5. New trajectory synthesized avoiding both initial zones and the new dynamic obstacle.
  6. Execution resumes; goal reached with zero collisions.

---

## 3. Quantitative Evaluation Metrics

| Metric | Mathematical Definition | Target / Unit |
|---|---|---|
| **Safety Violation Count** | $\sum_{h=0}^H \mathbf{1}\left(\exists m: b_m(\mathbf{x}_h) < 0\right)$ | 0 for SafeDiffuser |
| **Minimum Safety Margin** | $\min_{h \in [0, H]} \min_{m} b_m(\mathbf{x}_h)$ | $\ge 0.0$ (positive = safe) |
| **Collision Rate (%)** | $\frac{N_{\text{colliding}}}{N_{\text{total}}} \times 100\%$ | 0.0% |
| **Success Rate (%)** | Fraction reaching $\|\mathbf{x}_H - \mathbf{s}_{\text{goal}}\|_2 \le \epsilon$ without violation | $\ge 95\%$ |
| **Trajectory Length** | $\sum_{h=1}^H \|\mathbf{x}_h - \mathbf{x}_{h-1}\|_2$ | meters |
| **Planning Latency** | Time elapsed from API call to trajectory generation | milliseconds |
| **Replanning Latency** | Time elapsed from obstacle injection to safe trajectory deployment | $\le 250$ ms |

---

## 4. Performance Modes & Hardware Profiles

To ensure smooth operation on local hardware (RTX 4050 Laptop GPU, 6 GB VRAM, 16 GB RAM):

| Parameter | FAST_DEMO | BALANCED | RESEARCH |
|---|---|---|---|
| Diffusion Steps ($K$) | 15 | 30 | 100 |
| Trajectory Horizon ($H$) | 32 | 48 | 64 |
| CBF Active Steps | Last 10 steps ($k \le 10$) | Last 20 steps ($k \le 20$) | All steps ($k \le K$) |
| Solver Method | Analytical Closed-Form Dual | Analytical / CVXPY QP | Batched CVXPY QP |
| Target Latency | $< 80$ ms | $< 250$ ms | $< 800$ ms |
