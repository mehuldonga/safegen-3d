# SAFEGEN 3D — Step-by-Step Interactive Demo Guide

---

## 1. Launching the Prototype

Open a PowerShell terminal and run:
```powershell
.\scripts\start_dev.ps1
```
Open your web browser and navigate to:
**`http://localhost:5173`**

---

## 2. Navigating the 3D Interface

- **Rotate View:** Click and drag with the Left Mouse Button.
- **Pan View:** Click and drag with the Right Mouse Button.
- **Zoom In / Out:** Scroll the Mouse Wheel.
- **Reset Camera:** Click the **Reset View** button on the viewport toolbar.

---

## 3. Walkthrough Scenarios

### Scenario 1: Demonstrating the Safety Failure of Vanilla Diffusion
1. Look at the **Left Mission Panel**.
2. Set Start Position: `Y = 1.0`, `X = 1.0`.
3. Set Goal Position: `Y = 7.0`, `X = 7.0`.
4. Under **Planner Configuration**, select **Vanilla Diffuser**.
5. Click the green **GENERATE TRAJECTORY** button.
6. **Observation:**
   - A bright red path appears, connecting the beacons in a straight line that penetrates directly through the central red restricted zone.
   - Look at the **Bottom Metrics Bar**: Notice that **Safety Violations > 0** and **Min Margin < 0.0**.

### Scenario 2: Demonstrating SafeDiffuser Collision Avoidance
1. Under **Planner Configuration**, switch mode to **SafeDiffuser (CBF)**.
2. Select Performance Mode: **Fast Demo** or **Balanced**.
3. Click **GENERATE TRAJECTORY**.
4. **Observation:**
   - A vibrant glowing green path appears.
   - The path bends cleanly around the restricted zone perimeter.
   - Look at the **Bottom Metrics Bar**: **Safety Violations = 0** and **Min Margin > 0.0**.

### Scenario 3: The Hero Dynamic Obstacle Injection & Automatic Replanning
1. With the safe green trajectory active, click the blue **START EXECUTION** button.
2. Observe the blue robot marker begin moving smoothly along the green trajectory.
3. When the robot reaches ~40% of its journey, click the orange **INJECT OBSTACLE** button in the control panel.
4. **System Response:**
   - A new dynamic obstacle appears directly in front of the robot.
   - The simulation immediately halts execution and enters `REPLANNING` state.
   - The event feed logs `REPLAN_REQUESTED: Dynamic obstacle intersecting trajectory`.
   - SafeDiffuser replans a new detour path in amber from the robot's current coordinate.
   - The simulation automatically resumes along the amber path and navigates safely to the goal beacon!
