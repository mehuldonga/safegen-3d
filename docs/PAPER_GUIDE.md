# SAFEGEN 3D — Developer's Guide to the SafeDiffuser Paper

**Target Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)  
**Authors:** Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus  

---

## 1. Problem Statement & Motivation

Generative diffusion models have demonstrated remarkable empirical success in trajectory generation and offline reinforcement learning (e.g., Diffuser, Decision Diffuser). By treating trajectories $\boldsymbol{\tau} = (\mathbf{s}_0, \mathbf{a}_0, \dots, \mathbf{s}_H)$ as whole 2D sequences, diffusion models bypass the greedy shortsightedness and credit assignment bottlenecks of standard RL, effortlessly capturing multimodal behaviors (such as choosing whether to navigate left or right around an obstacle).

**However, ordinary diffusion models lack safety guarantees:**
1. Diffusion generates trajectories via stochastic Langevin dynamics or reverse score sampling. If the training dataset contains even marginally risky paths, the model assigns non-zero probability density across safety borders.
2. Under test-time distribution shifts (unseen start/goal pairs, newly moved obstacles), the unconstrained reverse drift routinely guides waypoints through forbidden zones.
3. Classifier-guided diffusion attempts to steer trajectories by adding a gradient $\nabla_{\boldsymbol{\tau}} \log p(\text{safe} \mid \boldsymbol{\tau})$. Because this gradient is a soft additive penalty, it trades off against the primary score matching term, frequently failing hard constraints when the goal-seeking score dominates.

---

## 2. Mathematical Intuition: How SafeDiffuser Enforces Safety

### 2.1 The Concept of Control Barrier Functions (CBFs)
In nonlinear control theory, a Control Barrier Function $b(\mathbf{x})$ defines a safe set $\mathcal{C} = \{\mathbf{x} \mid b(\mathbf{x}) \ge 0\}$.  
To ensure that state trajectories never leave $\mathcal{C}$ (forward set invariance), the derivative of $b$ along the system trajectory must satisfy:
$$\dot{b}(\mathbf{x}) \ge -\kappa \cdot b(\mathbf{x}), \quad \kappa > 0$$
This means that as the state approaches the boundary $\partial \mathcal{C}$ ($b \to 0$), the allowable velocity directed towards the obstacle decreases to zero. The state cannot cross into the unsafe region $b < 0$.

### 2.2 Reverse Diffusion as a Dynamical Control System
SafeDiffuser makes a brilliant theoretical leap: **It views the reverse denoising sequence $k = K \to 0$ as a discrete dynamical system.**
At each step $k$, the neural network proposes a nominal next iterate:
$$\hat{\mathbf{x}}^{k-1} = \boldsymbol{\mu}_\theta(\mathbf{x}^k, k) + \sigma_k \mathbf{z}$$
The step difference $\mathbf{u} = \mathbf{x}^{k-1} - \mathbf{x}^k$ is treated as a **control input**.  
Instead of blindly accepting the unconstrained proposal, SafeDiffuser filters $\mathbf{u}$ by solving a minimal-intervention Quadratic Program:
$$\min_{\mathbf{u}} \frac{1}{2} \|\mathbf{u} - \hat{\mathbf{u}}\|_2^2 \quad \text{subject to } \mathbf{G}(\mathbf{x}^k) \mathbf{u} \le \mathbf{h}(\mathbf{x}^k, k)$$
This minimally perturbs the diffusion model's proposal, preserving as much generative diversity and goal-directed intent as possible while strictly enforcing that $\mathbf{x}^{k-1}$ remains within the safe barrier.

### 2.3 Why Ordinary CBF Fails on Pure Noise: Time-Varying Relaxation (TVS)
At the start of reverse diffusion ($k = K$), the trajectory is pure random Gaussian noise $\boldsymbol{\tau}^K \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. Many points in this random noise cloud will naturally fall inside obstacle zones.  
If we rigidly enforced $b(\mathbf{x}) \ge 0$ at $k=K$, the filter would violently project pure noise against obstacle perimeters, ruining the global topological structure of the trajectory distribution.

To solve this, SafeDiffuser introduces **Time-Varying Diffusion Invariance (TVS)**:
$$B(\mathbf{x}, k) = b(\mathbf{x}) - \sigma(k_{\mathrm{bias}} - k)$$
- When $k$ is large (early noisy phase), $\sigma(k_{\mathrm{bias}} - k) \approx 0 \implies B \approx b$, meaning large negative margins are tolerated.
- As $k \to 0$ (final clean phase), $\sigma(k_{\mathrm{bias}} - k) \to 1 \implies b(\mathbf{x}) \ge 0$ is strictly guaranteed.
- Furthermore, upstream discovered that applying the CBF filter only during the final 10-20 steps ($k \le 10$) gives the fastest and most natural obstacle circumvention!

---

## 3. What Developers Must Know Before Modifying the Core

1. **Do not modify Lie derivatives without re-checking the power $p$:**
   - For circular/elliptical obstacles ($p=2$): $\nabla_{\mathbf{x}} b = \frac{2(x - x_c)}{r_x^2}$.
   - For super-ellipsoids ($p=4$): $\nabla_{\mathbf{x}} b = \frac{4(x - x_c)^3}{r_x^4}$.
2. **Coordinate convention:** In the 2D maze, states are structured as $[y, x, v_y, v_x]$ or $[x, y, v_x, v_y]$ depending on array indexing. Our implementation explicitly documents positional slices in `barrier_functions.py`.
3. **Dual KKT Solver Limits:** The closed-form analytical dual solver is derived for up to 2 concurrently active constraints. When navigating tight corridors with $> 2$ nearby obstacles, use the general `solve_qp_filter()` which handles arbitrary numbers of linear inequality constraints.
