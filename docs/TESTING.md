# SAFEGEN 3D — Verification & Testing Protocol

---

## 1. Overview of Test Architecture

The testing framework is structured across three complementary tiers:
1. **Machine Learning Unit Tests (`tests/ml/`):** Validates tensor dimensions, noise schedules, forward diffusion convergence, barrier evaluations, spatial Lie derivatives, and the analytical KKT dual solver.
2. **API & Contract Tests (`tests/api/`):** Validates FastAPI endpoints (`/health`, `/planning/create`, `/planning/generate`, `/simulation/obstacle`, `/metrics`) against valid and malformed payloads.
3. **Integration & Lifecycle Tests (`tests/integration/`):** Validates the full closed-loop event cycle (Plan $\to$ Execute $\to$ Dynamic Obstacle Injection $\to$ State Arrest $\to$ Replan $\to$ Resumption $\to$ Goal Completion).

---

## 2. Running the Test Suites

### All Tests via Automated Script:
```powershell
.\scripts\run_tests.ps1
```

### Targeted Module Execution:
```powershell
# Run ML Tests
python -m unittest tests/ml/test_diffusion.py
python -m unittest tests/ml/test_safety.py

# Run Integration Tests
python -m unittest tests/integration/test_replanning.py

# Run API Tests
python -m unittest tests/api/test_endpoints.py
```

---

## 3. Test Coverage Matrix

| Test Suite | File | Focus Areas | Criteria |
|---|---|---|---|
| **Temporal UNet & Schedule** | `test_diffusion.py` | Noise schedules, UNet forward shape, start/goal conditioning mask | Tensor shapes match `[B, H, D]`, zero NaNs |
| **Barrier & Lie Derivatives** | `test_safety.py` | Super-ellipsoid sign verification, gradient outward pointing, KKT dual repulsion | $b(\mathbf{x}) > 0$ outside, $b(\mathbf{x}) < 0$ inside |
| **Trajectory Validator** | `test_safety.py` | Post-generation safety validation & metric extraction | Rejects colliding trajectories, passes safe ones |
| **Replanning Event Loop** | `test_replanning.py` | World state transitions, event bus logging, replan counters | Emits `REPLAN_REQUESTED` and `REPLAN_COMPLETED` |
| **API Endpoints** | `test_endpoints.py` | REST response status codes, Pydantic schema validation | 200 on valid, 422 on invalid, 503 on unready |
