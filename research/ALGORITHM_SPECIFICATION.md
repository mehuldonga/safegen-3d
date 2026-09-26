# SAFEGEN 3D — Algorithm Specification

**Reference Paper:** *SafeDiffuser: Safe Planning with Diffusion Probabilistic Models* (ICLR 2025)

---

## Algorithm 1: Training the Temporal UNet Score Estimator (DDPM)

```python
Input: Offline trajectory dataset D = {τ^(i)}, total training steps S, batch size B
Output: Optimized score network ε_θ(τ, k)

1: Initialize Temporal UNet parameters θ, noise schedule β_1, ..., β_K
2: for step = 1 to S do:
3:     Sample mini-batch of clean trajectories τ^0 ~ D
4:     Sample diffusion timesteps k ~ Uniform({1, ..., K}) for each sample in batch
5:     Sample standard Gaussian noise ε ~ N(0, I)
6:     Compute noisy trajectory:
           τ^k = sqrt(ᾱ_k) * τ^0 + sqrt(1 - ᾱ_k) * ε
7:     Apply condition masking (retain start state s_0 and goal state s_H):
           τ^k = apply_conditioning(τ^k, conditions)
8:     Compute prediction: ε_pred = ε_θ(τ^k, k)
9:     Compute loss: L(θ) = MSE(ε_pred, ε)
10:    Update θ via Adam optimizer: θ ← θ - η ∇_θ L(θ)
11: end for
12: return ε_θ
```

---

## Algorithm 2: SafeDiffuser Reverse Sampling with CBF Filter (Inference)

```python
Input: Score network ε_θ, start state s_0, goal state s_H, horizon H,
       obstacles O = {O_1, ..., O_M}, noise schedule {α_k, β_k, ᾱ_k}
Output: Safe trajectory τ^0 satisfying ∀ h, τ_h^0 ∉ O

1:  Initialize τ^K ~ N(0, I) of shape [B, H, D]
2:  Enforce boundary conditions:
        τ^K[:, 0, :] = s_0,  τ^K[:, H-1, :] = s_H
3:  for k = K down to 1 do:
4:      # 1. Compute nominal DDPM / DDIM reverse proposal
5:      Compute model mean μ_θ(τ^k, k) and log variance
6:      Sample z ~ N(0, I) if k > 1 else z = 0
7:      Nominal candidate:
            τ_hat^{k-1} = μ_θ(τ^k, k) + σ_k * z
8:      Enforce boundary conditions on τ_hat^{k-1}
9:
10:     # 2. Apply SafeDiffuser Invariance Filter
11:     if mode == "SAFEDIFFUSER" then:
12:         for trajectory step h = 1 to H-2 do:  # internal waypoints
13:             x_curr = τ^k[h]
14:             x_nom = τ_hat^{k-1}[h]
15:             u_ref = x_nom - x_curr
16:
17:             # Evaluate barrier functions and gradients for all obstacles
18:             for each obstacle m = 1 to M do:
19:                 b_m = compute_barrier(x_curr, O_m, k)
20:                 G_m = -∇_x b_m(x_curr)
21:                 h_m = κ * b_m + ∂b_m/∂k
22:             end for
23:
24:             # Solve Quadratic Program or Closed-Form Dual
25:             u_safe = solve_cbf_qp(u_ref, G, h)
26:             τ^{k-1}[h] = x_curr + u_safe
27:         end for
28:     else:  # Vanilla mode
29:         τ^{k-1} = τ_hat^{k-1}
30:     end if
31:
32:     # 3. Re-enforce conditioning
33:     τ^{k-1}[:, 0, :] = s_0
34:     τ^{k-1}[:, H-1, :] = s_H
35: end for
36: return τ^0
```

---

## Algorithm 3: Closed-Loop Dynamic Replanning & Digital Twin Monitoring

```python
Input: Target goal s_goal, robot controller, SafeDiffuser planner, update frequency f
Output: Successful mission completion or verified failure report

1:  state = robot.get_current_state()
2:  active_trajectory = planner.generate(start=state, goal=s_goal, obstacles=world.obstacles)
3:  world.set_active_trajectory(active_trajectory)
4:  step_idx = 0
5:
6:  while not world.is_goal_reached(state, s_goal) do:
7:      # 1. Check for dynamic environmental changes (Obstacle Injection)
8:      if world.has_obstacle_changed() or world.is_path_compromised(active_trajectory, step_idx) then:
9:          world.log_event(REPLAN_REQUESTED, reason="Dynamic obstacle intersecting current trajectory")
10:         world.pause_execution()
11:
12:         # 2. Trigger real-time replanning from current robot state
13:         new_trajectory = planner.generate(
14:             start=robot.get_current_state(),
15:             goal=s_goal,
16:             obstacles=world.obstacles,
17:             mode="SAFEDIFFUSER"
18:         )
19:
20:         # 3. Validate new trajectory
21:         is_valid, margin = validate_trajectory_safety(new_trajectory, world.obstacles)
22:         if is_valid then:
23:             world.log_event(REPLAN_COMPLETED, margin=margin)
24:             world.set_active_trajectory(new_trajectory)
25:             step_idx = 0
26:             world.resume_execution()
27:         else:
28:             world.log_event(PLAN_REJECTED, reason="No safe trajectory exists under constraints")
29:             world.stop_execution()
30:             return FAILURE
31:         end if
32:     end if
33:
34:     # 4. Execute next step in simulation
35:     robot.step_towards(active_trajectory[step_idx])
36:     step_idx += 1
37:     sleep(1/f)
38: end while
39: world.log_event(EXECUTION_COMPLETED)
40: return SUCCESS
```
