# SAFEGEN 3D — 3–5 Minute Presentation & Live Demo Script

**Presenter Target:** AI Researchers, Robotics Engineers, Executive Leadership  
**Application URL:** `http://localhost:5173`  
**Total Runtime:** ~4 Minutes  

---

## ⏱️ Minute 0:00 - 0:45 | Introduction & Research Motivation

**Action:** Open browser to `http://localhost:5173`. Show the 3D Digital Twin environment.

**Narration:**
> "Welcome to **SAFEGEN 3D**. Today we are demonstrating an end-to-end realization of **SafeDiffuser**, a breakthrough trajectory planning framework presented at **ICLR 2025** by MIT CSAIL and IBM Research.
> 
> Diffusion probabilistic models are revolutionary for motion planning because they generate diverse, multimodal trajectory distributions from offline data. But here is the critical flaw: standard diffusion models have **zero safety guarantees**. In complex environments with restricted zones or dynamic obstacles, ordinary diffusion planners regularly generate paths that slice directly through hazards.
> 
> SafeDiffuser solves this by introducing **Control Barrier Functions (CBFs)** and **Finite-Time Diffusion Invariance** directly into the reverse denoising process. Let's see this in action."

---

## ⏱️ Minute 0:45 - 1:45 | Baseline vs. SafeDiffuser Comparison

**Action:**
1. In the Left Mission Panel, select Start `[1.0, 1.0]` and Goal `[7.0, 7.0]`. Note the red restricted zone at `[4.0, 4.0]`.
2. Select **Planner Mode: Vanilla Diffuser**. Click **GENERATE TRAJECTORY**.
3. Point out the red trajectory line in the 3D canvas and the bottom metrics.

**Narration:**
> "First, let's run the unconstrained **Vanilla Diffuser**. Notice what happens: the generative model seeks the shortest path to the goal and drafts a path cutting straight through the red restricted zone. In the bottom bar, our safety validator logs multiple safety violations and a negative minimum safety margin. If executed, a real robot would breach this boundary.
> 
> Now, let's toggle to **SafeDiffuser** with our Time-Varying Barrier Filter and click **GENERATE TRAJECTORY**."

**Action:**
1. Switch to **SafeDiffuser**. Click **GENERATE TRAJECTORY**.
2. Rotate camera in 3D canvas with left mouse click to show the green trajectory cleanly curving around the restricted zone.

**Narration:**
> "Watch the glowing green trajectory. Without retraining the base neural network, the CBF safety filter evaluated the barrier function and Lie derivatives at each reverse diffusion step, repelling the trajectory outside the hazard set. In our bottom metrics: **Zero safety violations**, with a strictly positive safety margin."

---

## ⏱️ Minute 1:45 - 3:00 | The Hero Demonstration: Dynamic Obstacle & Automatic Replanning

**Action:**
1. Click **START EXECUTION**.
2. Watch the blue robot marker begin moving along the green planned path.
3. At ~40% progress, click the bold orange **INJECT OBSTACLE** button.
4. Watch the simulation pause, the event log register `REPLAN_REQUESTED`, the amber replanned trajectory appear, and execution resume.

**Narration:**
> "Now for our hero scenario: dynamic closed-loop execution. The robot begins traversing its safe plan. 
> 
> Suddenly, an unexpected obstacle appears directly on the robot's future path!
> 
> Notice what our digital twin did:
> 1. It detected the spatial collision hazard immediately.
> 2. It arrested the robot's velocity to prevent collision.
> 3. It automatically triggered SafeDiffuser replanning using the robot's current coordinate as the new start condition.
> 4. In less than 150 milliseconds, our analytical KKT solver computed a safe detour around the newly injected hazard—shown here in glowing amber.
> 5. The robot resumed navigation and reached the goal with 100% mission success and zero collisions!"

---

## ⏱️ Minute 3:00 - 4:00 | Research Traceability & Architectural Wrap-Up

**Action:** Open `docs/REQUIREMENTS_MATRIX.md` or show the Right AI Status Panel with QP solver stats.

**Narration:**
> "To conclude, what distinguishes SAFEGEN 3D is our strict adherence to research integrity:
> - **From the Paper:** The Temporal UNet architecture, the super-ellipsoid Control Barrier Functions, the time-varying sigmoid relaxation, and the closed-form KKT dual solver are all directly ported from SafeDiffuser ICLR 2025.
> - **Our Product Layer:** We engineered the real-time React + Three.js 3D digital twin, the FastAPI WebSocket streaming architecture, the self-contained kinematic simulation, the dynamic obstacle replanner, and the autonomous requirement verification agent.
> 
> Thank you, and we welcome any technical questions!"
