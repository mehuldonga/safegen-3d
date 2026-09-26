# SafeDiffuser: Safe Planning with Diffusion Probabilistic Models — Comprehensive Paper Analysis

**Conference:** International Conference on Learning Representations (ICLR) 2025  
**Authors:** Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus (MIT CSAIL, IBM Research)  
**Primary Paper:** [ICLR 2025 Conference Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f95606d8e870020085990d9650b4f2a1-Paper-Conference.pdf)  
**Project Page:** https://safediffuser.github.io/safediffuser/  
**Official Code:** https://github.com/Weixy21/SafeDiffuser  

---

## 1. Executive Summary

SafeDiffuser addresses a fundamental vulnerability in generative diffusion-based decision making: while diffusion models (such as Janner et al.'s *Diffuser* and Ajay et al.'s *Decision Diffuser*) excel at synthesizing multimodal, long-horizon trajectory distributions from offline datasets, they offer **no formal safety guarantees**. Consequently, generated plans frequently cross hazard regions, collide with static/dynamic obstacles, or breach state-space bounds—particularly when conditioned on out-of-distribution start-goal pairs or deployed in safety-critical settings.

SafeDiffuser solves this by introducing **Control Barrier Functions (CBFs)** and **Finite-Time Diffusion Invariance** directly into the reverse stochastic/deterministic denoising process. Rather than relying on unguided post-processing or soft classifier guidance penalties (which easily fail to satisfy hard safety constraints), SafeDiffuser formulates the denoising step as a sequence of **Safety-Constrained Quadratic Programs (QPs)** or analytical projections. This guarantees that trajectory iterates remain inside the safe set $\mathcal{C}$ at every diffusion step or converge to safety at $t=0$.

---

## 2. Core Problem Definition

### 2.1 Why Planning with Diffusion?
Traditional trajectory optimization (e.g., CHOMP, TrajOpt, MPC) suffers from local minima in non-convex environments and scales poorly with planning horizon. Reinforcement learning methods struggle with long-horizon credit assignment and multimodality.

Diffusion models reframe trajectory planning as conditional generative modeling:
- A trajectory $\boldsymbol{\tau} = (\mathbf{s}_0, \mathbf{a}_0, \mathbf{s}_1, \mathbf{a}_1, \dots, \mathbf{s}_H)$ of horizon $H$ is treated as an image-like $2\mathrm{D}$ array.
- Forward process adds Gaussian noise over diffusion timesteps $k \in \{1, \dots, K\}$.
- Reverse process denoises pure noise $\boldsymbol{\tau}^K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ conditioned on start state $\mathbf{s}_0$ and goal state $\mathbf{s}_H$:
  $$p_\theta(\boldsymbol{\tau}^{k-1} \mid \boldsymbol{\tau}^k, \mathbf{y}) = \mathcal{N}\left(\boldsymbol{\mu}_\theta(\boldsymbol{\tau}^k, k, \mathbf{y}), \mathbf{\Sigma}_\theta(\boldsymbol{\tau}^k, k)\right)$$
This global generation naturally captures multimodality (e.g., circumventing an obstacle from left or right) without suffering from myopic horizon truncation.

### 2.2 Why Can Ordinary Diffusion Planning Generate Unsafe Plans?
1. **Distributional Shift & Compounding Noise:** The offline dataset often contains marginally safe or imperfect trajectories. Stochastic sampling can place probability mass across constraint boundaries.
2. **Soft Conditioning Inadequacy:** Classifier-guided diffusion adds a gradient term $\nabla_{\boldsymbol{\tau}} \log p(\text{safe} \mid \boldsymbol{\tau})$ to the mean. Because this is an unconstrained soft penalty, the optimizer can sacrifice safety whenever the score matching gradient dominates.
3. **Black-Box Denoising:** Standard UNet denoisers have no intrinsic awareness of physical bounds, obstacles, or safety set invariance.
4. **Out-of-Distribution Start-Goal Pairs:** When querying tasks unseen in training, the unconditional prior can drag the trajectory through obstacles lying between start and goal.

### 2.3 Mathematical Meaning of "Safe"
Let the system state at trajectory step $h \in \{0, \dots, H\}$ be $\mathbf{x}_h \in \mathcal{X} \subset \mathbb{R}^n$.  
A safe set $\mathcal{C} \subset \mathcal{X}$ is defined as the 0-superlevel set of a continuously differentiable safety function $b: \mathcal{X} \to \mathbb{R}$:
$$\mathcal{C} = \{\mathbf{x} \in \mathcal{X} \mid b(\mathbf{x}) \ge 0\}$$
$$\partial \mathcal{C} = \{\mathbf{x} \in \mathcal{X} \mid b(\mathbf{x}) = 0\}$$
$$\operatorname{Int}(\mathcal{C}) = \{\mathbf{x} \in \mathcal{X} \mid b(\mathbf{x}) > 0\}$$

A trajectory $\boldsymbol{\tau} = (\mathbf{x}_0, \dots, \mathbf{x}_H)$ is **strictly safe** if and only if:
$$\forall h \in \{0, \dots, H\}, \quad \mathbf{x}_h \in \mathcal{C} \iff b(\mathbf{x}_h) \ge 0$$
For $M$ obstacles or safety constraints, the composite safe set is $\mathcal{C} = \bigcap_{m=1}^M \mathcal{C}_m = \{\mathbf{x} \mid \min_{m} b_m(\mathbf{x}) \ge 0\}$.

---

## 3. Mathematical Foundations of SafeDiffuser

### 3.1 Continuous Reverse SDE / ODE Denoising
Diffusion models approximate the reverse SDE (Song et al., 2020):
$$\mathrm{d}\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] \mathrm{d}t + g(t) \mathrm{d}\bar{\mathbf{w}}$$
In discrete time with DDPM/DDIM sampling, given current iterate $\mathbf{x}^k$ at diffusion step $k$, the unconstrained denoiser proposes a candidate next state:
$$\hat{\mathbf{x}}^{k-1} = \boldsymbol{\mu}_\theta(\mathbf{x}^k, k) + \sigma_k \boldsymbol{\epsilon}$$

### 3.2 Control Barrier Functions for Diffusion Invariance
SafeDiffuser treats the reverse diffusion process itself as a dynamic system indexed by reverse diffusion time $t \in [0, T]$ (where $t$ goes from $T \to 0$ or index $k$ goes from $K \to 0$).

To guarantee that the generated trajectory $\mathbf{x}^0$ resides in $\mathcal{C}$, SafeDiffuser develops three variants of **Diffusion Invariance**:

#### 1. Robust Safe Diffuser (RoS)
Enforces safety invariance at every single reverse step $k$. If the initial sample $\mathbf{x}^K \notin \mathcal{C}$, it projects iterates towards $\mathcal{C}$ using the discrete CBF condition:
$$\Delta b(\mathbf{x}^k, \mathbf{u}^k) \ge -\gamma b(\mathbf{x}^k), \quad \gamma \in (0, 1]$$
where $\mathbf{u}^k = \mathbf{x}^{k-1} - \mathbf{x}^k$ is the reverse diffusion step control input.

#### 2. Relaxed Safe Diffuser (ReS)
Recognizes that requiring intermediate noisy states $\mathbf{x}^K, \mathbf{x}^{K-1}, \dots$ (which represent pure Gaussian noise or heavily degraded samples) to satisfy physical obstacle avoidance is overly restrictive and can disrupt generative fidelity.
ReS introduces a time-dependent relaxation factor $\alpha(k)$ such that:
$$b(\mathbf{x}^{k-1}) \ge (1 - \alpha(k)) b(\mathbf{x}^k) - \beta(k)$$
where $\alpha(k) \to 1$ and $\beta(k) \to 0$ as $k \to 0$.

#### 3. Time-Varying Safe Diffuser (TVS)
Formulates a time-varying barrier function $B(\mathbf{x}, k) = b(\mathbf{x}) - \sigma(k_{\mathrm{bias}} - k)$ where $\sigma(\cdot)$ is the sigmoid function:
- When $k$ is large (initial noisy stages), $\sigma(k_{\mathrm{bias}} - k) \approx 0$, allowing the diffusion model full freedom to explore global topology.
- When $k \to 0$ (final clean stages), $\sigma(k_{\mathrm{bias}} - k) \to 1$, enforcing the exact barrier constraint $b(\mathbf{x}) \ge 0$.

### 3.3 QP Safety Filter Formulation
At each reverse diffusion step $k$, let $\hat{\mathbf{x}}^{k-1}$ be the nominal proposal from the trained diffusion model $\boldsymbol{\mu}_\theta$. SafeDiffuser computes the safe iterate $\mathbf{x}^{k-1}$ via the following Quadratic Program:

$$\mathbf{x}^{k-1} = \arg\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{x} - \hat{\mathbf{x}}^{k-1}\|_2^2$$
$$\text{subject to: } \mathbf{G}_m(\mathbf{x}^k) (\mathbf{x} - \mathbf{x}^k) \le \mathbf{h}_m(\mathbf{x}^k, k), \quad \forall m \in \{1, \dots, M\}$$

Where:
$$\mathbf{G}_m(\mathbf{x}^k) = -\nabla_{\mathbf{x}} b_m(\mathbf{x}^k)$$
$$\mathbf{h}_m(\mathbf{x}^k, k) = \kappa \cdot b_m(\mathbf{x}^k) + \frac{\partial b_m}{\partial t}$$

### 3.4 Closed-Form Analytical Solution for Real-Time Execution
In environments with multiple obstacles, solving batch QPs with interior-point methods can be computationally intensive. SafeDiffuser provides an analytical dual closed-form solution (`invariance_time_cf`) derived via KKT optimality conditions for active constraints:
Given dual variables $\lambda_1^*, \lambda_2^*$ for the critical obstacles:
$$\mathbf{x}^{k-1} = \hat{\mathbf{x}}^{k-1} + \sum_{m} \lambda_m^* \mathbf{G}_m^\top$$
This closed-form formulation runs in microseconds per diffusion step, making real-time dynamic replanning viable on local hardware.

---

## 4. Evaluation Tasks in the Paper

1. **Maze2D (Navigation under hard obstacles):**
   - Agent must navigate from start to target without penetrating circular/super-ellipsoidal forbidden zones.
   - Evaluated on collision rates, trajectory length, and ELBO.
2. **Legged Robot Locomotion:**
   - Multi-joint quadrupeds avoiding hazard areas while maintaining dynamic balance.
3. **3D Space Robotic Manipulation (KUKA Arm):**
   - End-effector path generation avoiding restricted volumes in 3D workspace.

---

## 5. Paper Limitations and Adaptations

| Aspect | Paper Specification | Upstream Codebase | Local SAFEGEN 3D Implementation |
|---|---|---|---|
| Simulation Engine | MuJoCo 200 + D4RL (Linux only) | Requires `.mujoco/mujoco200/bin` | **Self-contained deterministic Maze2D & 3D Digital Twin** (Windows compatible) |
| Diffusion Model | Temporal UNet with 1D Convolutions | `diffuser/models/temporal.py` | Full PyTorch Temporal UNet ported with identical architecture |
| Safety Solver | `qpth` batched QP & analytical closed-form | `diffuser/models/diffusion.py:830-1030` | Dual solver: `cvxpy`/analytical closed-form TVS filter |
| Execution & Replanning | Open-loop evaluation (plan once at $t=0$) | `scripts/plan_maze2d.py` | **Closed-loop Digital Twin with Dynamic Obstacle Injection & Automatic Replanning** |
| User Interface | Offline matplotlib 2D png dumps | None (script output only) | **Full Interactive 3D Digital Twin (React + Three.js)** |

---

## 6. Scientific Integrity Notice
In accordance with Non-Negotiable Source of Truth:
- The CBF formulation, Lie derivatives, and TVS equations are directly taken from SafeDiffuser ICLR 2025.
- The web UI, real-time WebSocket communication, and dynamic replanning loop are **[ENGINEERING ADDITIONS]** built as a product layer around the verified research core.
- The removal of the Linux-only MuJoCo dependency in favor of a standalone kinematic simulator is an **[ADAPTATION FOR LOCAL HARDWARE]**.
