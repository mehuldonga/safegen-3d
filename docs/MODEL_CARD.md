# SAFEGEN 3D — Model Card: SafeDiffuser Temporal UNet

---

## 1. Model Details

- **Model Name:** SafeDiffuser Temporal UNet Denoising Score Model
- **Version:** 1.0 (Adapted for local edge execution)
- **Architecture:** 1D Temporal Convolutional Residual UNet with Mish activation and GroupNorm.
- **Developers:** Adapted from Xiao et al. (MIT CSAIL & IBM Research, ICLR 2025).
- **Primary Task:** Reverse denoising of continuous trajectory state sequences $\boldsymbol{\tau} \in \mathbb{R}^{H \times D}$.
- **Input Dimension:** Horizon $H=48$, Transition Dimension $D=4$ ($[y, x, v_y, v_x]$).
- **Conditioning Mechanism:** Direct boundary state in-painting mask for start state $\mathbf{s}_0$ and goal state $\mathbf{s}_H$.

---

## 2. Intended Use

- **Primary Intended Use:** Trajectory planning for mobile ground robots navigating cluttered 2D/3D workspaces with convex and non-convex obstacles.
- **Downstream Applications:** Autonomous warehouse mobile robots (AMRs), factory floor navigation, robotic manipulation corridor clearance.
- **Out-of-Scope Uses:** High-frequency flight control, highly chaotic turbulence fields, unconstrained open-world perception without safety barrier definitions.

---

## 3. Safety Mechanism & Control Barrier Functions

Unlike standard generative models whose safety relies on fine-tuning or post-hoc trajectory rejection:
1. **Control Barrier Function Integration:** Enforces safety at each reverse denoising step $k \in \{1, \dots, K\}$.
2. **Time-Varying Relaxation (TVS):** Dynamically scales constraint firmness using a sigmoid schedule $\sigma(k_{\mathrm{bias}} - k)$, preserving generative multimodal exploration during high-noise iterations while ensuring strict barrier satisfaction as iterates emerge into clean states ($k \to 0$).
3. **Solver Guarantees:** Solves a Quadratic Program $\min \frac{1}{2}\|\mathbf{u} - \hat{\mathbf{u}}\|^2$ subject to Lie derivative inequalities $\mathbf{G} \mathbf{u} \le \mathbf{h}$.

---

## 4. Hardware Efficiency & Footprint

- **Parameter Count:** ~1.2M parameters (optimized for local consumer GPUs).
- **VRAM Footprint:** $< 1.5$ GB under active inference.
- **Inference Latency:**
  - `FAST_DEMO` mode (15 steps, analytical KKT): $\sim 45$ ms on RTX 4050.
  - `BALANCED` mode (30 steps, analytical KKT): $\sim 95$ ms on RTX 4050.
  - `RESEARCH` mode (100 steps, CVXPY QP): $\sim 450$ ms on RTX 4050.
