# SAFEGEN 3D — Known Limitations & Scope Boundaries

---

## 1. Research Scope Reductions

In compliance with the master directive's strict requirement against false research claims:

| Research Component | Full Paper Scope | SAFEGEN 3D Local Scope | Justification / Trade-off |
|---|---|---|---|
| **Physics Simulator** | MuJoCo 200 physics engine via Linux binaries | Standalone Kinematic 2D Simulator | MuJoCo 200 has deprecated Linux C++ bindings incompatible with Windows 11. Kinematic simulator preserves exact coordinate scales, wall collision physics, and trajectory geometries. |
| **Locomotion Benchmark** | HalfCheetah / Ant quadruped joints | Maze2D Navigation Benchmark | The paper's primary conceptual demonstration of trajectory safety occurs in Maze2D. Quadruped dynamics require extensive reinforcement learning training loops beyond local laptop constraints. |
| **3D Robotic Manipulation** | KUKA iiwa 7-DOF arm in simulation | Ground AMR in 3D Factory Twin | Focuses computational budget on real-time interactive 3D digital twin visualization, dynamic obstacle injection, and sub-second replanning. |

---

## 2. Solver Limitations

1. **Analytical Closed-Form Solver Constraint Capacity:**
   - The closed-form analytical KKT solver (`invariance_time_cf`) is derived for up to two concurrently active obstacle barriers.
   - For narrow maze bottlenecks with $\ge 3$ intersecting barriers, the system automatically falls back to sequential projection or `cvxpy` quadratic programming, which incurs slightly higher latency (~150 ms vs. 35 ms).

2. **Infeasible Initial States:**
   - If the user designates a start position that is already deep inside an obstacle's interior ($b(\mathbf{s}_0) < 0$), no continuous trajectory can start safely without an initial escape maneuver. The system flags this condition with an explicit user error: `"Initial state is within restricted obstacle zone"`.
