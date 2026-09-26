# SAFEGEN 3D — Mathematical Specification & Theoretical Foundations

**Reference Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)  
**Primary Authors:** Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus  

---

## 1. System Mathematical Overview

SafeDiffuser operates at the intersection of:
1. Continuous-time stochastic diffusion processes (forward SDE and reverse SDE/ODE).
2. Discrete-time score-based trajectory generation.
3. Nonlinear Control Barrier Functions (CBFs) ensuring forward set invariance.
4. Convex optimization (Quadratic Programming) projecting diffusion drift toward the safe manifold.

```
       Forward Process (Noising):
       τ^0 ~ q(τ) ───► τ^1 ───► ... ───► τ^K ~ N(0, I)

       Reverse Process with SafeDiffuser (Denoising + CBF Invariance):
       τ^K ~ N(0, I) ───► ... ───► [ Nominal Drift μ_θ ]
                                             │
                                             ▼
                                  [ CBF Safety Filter ]
                                  (min ||τ - μ_θ||² s.t. G·Δτ ≤ h)
                                             │
                                             ▼
                                    τ^{k-1} ∈ C(k) ───► ... ───► τ^0 ∈ C_safe
```

---

## 2. Core Equation Catalog & Traceability

### Equation 1: Discrete Forward Noising (DDPM Schedule)
$$\boldsymbol{\tau}^k = \sqrt{\bar{\alpha}_k} \boldsymbol{\tau}^0 + \sqrt{1 - \bar{\alpha}_k} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- **Variables:**
  - $\boldsymbol{\tau} \in \mathbb{R}^{H \times D}$: Trajectory array over horizon $H$, dimension $D = d_s + d_a$.
  - $k \in \{1, \dots, K\}$: Diffusion step index ($K=100$ or $K=20$).
  - $\beta_k \in [\beta_1, \beta_K]$: Linear or cosine variance schedule.
  - $\alpha_k = 1 - \beta_k$, $\bar{\alpha}_k = \prod_{s=1}^k \alpha_s$.
- **Paper Reference:** Section 3.1, Equation (1)-(2).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:270-310`.
- **Our Implementation:** `app/ml/diffusion/gaussian_diffusion.py:q_sample()`.
- **Classification:** `[SPECIFIED]`

---

### Equation 2: Reverse Denoising Drift & Dispersion
$$\hat{\boldsymbol{\tau}}^{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \boldsymbol{\tau}^k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}} \boldsymbol{\epsilon}_\theta(\boldsymbol{\tau}^k, k) \right) + \sigma_k \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- **Variables:**
  - $\boldsymbol{\epsilon}_\theta$: Parameterized Temporal UNet neural network predicting added noise.
  - $\sigma_k = \sqrt{\tilde{\beta}_k} = \sqrt{\frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k} \beta_k}$.
  - $\hat{\boldsymbol{\tau}}^{k-1}$: Unconstrained nominal next iterate proposal.
- **Paper Reference:** Section 3.1, Equation (3).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:1030-1039`.
- **Our Implementation:** `app/ml/diffusion/gaussian_diffusion.py:p_sample()`.
- **Classification:** `[SPECIFIED]`

---

### Equation 3: Continuous & Discrete Control Barrier Function (CBF)
Let the safety barrier function for obstacle $m$ be $b_m(\mathbf{x}): \mathbb{R}^2 \to \mathbb{R}$. The safe set is $\mathcal{C}_m = \{\mathbf{x} \mid b_m(\mathbf{x}) \ge 0\}$.
For dynamic safety invariance under reverse step $\Delta \mathbf{x}^k = \mathbf{x}^{k-1} - \mathbf{x}^k$:
$$\Delta b_m(\mathbf{x}^k) \approx \nabla_{\mathbf{x}} b_m(\mathbf{x}^k)^\top \Delta \mathbf{x}^k \ge -\kappa_k b_m(\mathbf{x}^k)$$
Rearranging into standard inequality form $\mathbf{G}_m \Delta \mathbf{x}^k \le h_m$:
$$\mathbf{G}_m = -\nabla_{\mathbf{x}} b_m(\mathbf{x}^k)^\top$$
$$h_m = \kappa_k b_m(\mathbf{x}^k) + \frac{\partial b_m}{\partial k}$$
- **Paper Reference:** Section 4.1, Theorem 1 & Equation (8)-(11).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:830-860`.
- **Our Implementation:** `app/ml/safety/barrier_functions.py` & `cbf_filter.py`.
- **Classification:** `[SPECIFIED]`

---

### Equation 4: Super-Ellipsoidal Barrier Formulations
For an obstacle centered at $(x_c, y_c)$ with semi-axes $(r_x, r_y)$ and order $p \in \{2, 4\}$:
$$b_m(x, y) = \left( \frac{y - y_c}{r_y} \right)^p + \left( \frac{x - x_c}{r_x} \right)^p - 1 - \delta$$
where $\delta > 0$ is a safety margin buffer (typically $\delta \in [0.01, 0.4]$).
- **Lie / Spatial Derivatives:**
  $$\frac{\partial b_m}{\partial y} = p \left( \frac{y - y_c}{r_y} \right)^{p-1} \frac{1}{r_y}$$
  $$\frac{\partial b_m}{\partial x} = p \left( \frac{x - x_c}{r_x} \right)^{p-1} \frac{1}{r_x}$$
- **Paper Reference:** Section 5.1 (Maze Navigation Experiment).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:830-855`.
- **Our Implementation:** `app/ml/safety/barrier_functions.py:SuperEllipsoidBarrier`.
- **Classification:** `[SPECIFIED]`

---

### Equation 5: Time-Varying Safe Diffuser (TVS) Relaxation
$$B_m(\mathbf{x}, k) = b_m(\mathbf{x}) - \sigma(k_{\mathrm{bias}} - k) - \delta$$
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\frac{\partial B_m}{\partial k} = \frac{\mathrm{d} \sigma}{\mathrm{d} z} \frac{\mathrm{d} z}{\mathrm{d} k} = -\sigma(k_{\mathrm{bias}} - k)(1 - \sigma(k_{\mathrm{bias}} - k)) \cdot (-1) = \sigma(k_{\mathrm{bias}} - k)(1 - \sigma(k_{\mathrm{bias}} - k))$$
- **Variables:**
  - $k_{\mathrm{bias}} = 5$ or $50$ (transition midpoint).
  - $L_f b = \sigma(k_{\mathrm{bias}} - k)(1 - \sigma(k_{\mathrm{bias}} - k))$.
- **Paper Reference:** Section 4.3 (Time-Varying Diffusion Invariance).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:910-915`.
- **Our Implementation:** `app/ml/safety/cbf_filter.py:compute_tvs_barrier()`.
- **Classification:** `[SPECIFIED]`

---

### Equation 6: Quadratic Program (QP) Safety Filter
At diffusion step $k$, given nominal unconstrained update proposal $\hat{\mathbf{u}} = \hat{\mathbf{x}}^{k-1} - \mathbf{x}^k$:
$$\mathbf{u}^* = \arg\min_{\mathbf{u} \in \mathbb{R}^2} \frac{1}{2} \|\mathbf{u} - \hat{\mathbf{u}}\|_2^2$$
$$\text{subject to: } \mathbf{G}_m \mathbf{u} \le h_m, \quad \forall m \in \mathcal{A}(\mathbf{x}^k)$$
where $\mathcal{A}(\mathbf{x}^k)$ is the active index set of obstacles.
Then the safe step is:
$$\mathbf{x}^{k-1} = \mathbf{x}^k + \mathbf{u}^*$$
- **Paper Reference:** Section 4.2, Equation (14).
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:880-891`.
- **Our Implementation:** `app/ml/safety/cbf_filter.py:solve_qp_filter()`.
- **Classification:** `[SPECIFIED]`

---

### Equation 7: Analytical Closed-Form KKT Dual Solution
For two dominant active constraints with gradients $\mathbf{y}_1 = \mathbf{G}_1^\top, \mathbf{y}_2 = \mathbf{G}_2^\top$ and nominal command $\bar{\mathbf{u}} = \hat{\mathbf{u}}$:
$$p_1 = h_1 - \mathbf{G}_1 \bar{\mathbf{u}}, \quad p_2 = h_2 - \mathbf{G}_2 \bar{\mathbf{u}}$$
Gram matrix elements:
$$\Gamma_{11} = \|\mathbf{y}_1\|^2, \quad \Gamma_{12} = \mathbf{y}_1^\top \mathbf{y}_2, \quad \Gamma_{22} = \|\mathbf{y}_2\|^2$$
Dual multipliers $\lambda_1, \lambda_2 \ge 0$ are solved analytically:
$$\lambda_1^* = \begin{cases} 0 & \text{if } \Gamma_{12} [p_2]_- < \Gamma_{22} p_1 \\ \frac{[p_1]_-}{\Gamma_{11}} & \text{if } \Gamma_{12} [p_1]_- < \Gamma_{11} p_2 \\ \frac{\Gamma_{22} p_1 - \Gamma_{12} p_2}{\det(\boldsymbol{\Gamma})} & \text{otherwise} \end{cases}$$
Optimal safe action:
$$\mathbf{u}^* = \bar{\mathbf{u}} + \lambda_1^* \mathbf{y}_1 + \lambda_2^* \mathbf{y}_2$$
- **Paper Reference:** Appendix C & Section 4.3.
- **Official Repo Implementation:** `upstream/SafeDiffuser/diffuser/models/diffusion.py:960-1025`.
- **Our Implementation:** `app/ml/safety/cbf_filter.py:solve_closed_form_tvs()`.
- **Classification:** `[SPECIFIED]`
