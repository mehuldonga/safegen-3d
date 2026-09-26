# SAFEGEN 3D — Reproduction Notes & Audit

For complete upstream forensics and line-by-line code inspections, please consult:
👉 **[research/REPRODUCTION_NOTES.md](../research/REPRODUCTION_NOTES.md)**

---

## Executive Audit Summary

1. **Upstream Repository Preservation:**
   - The upstream repository `https://github.com/Weixy21/SafeDiffuser` is cloned and preserved in its pristine state under `upstream/SafeDiffuser/`.
   - No upstream files were modified, overwritten, or obscured.

2. **Core Algorithm Integrity:**
   - Reverse SDE discretization, Temporal UNet layers, and Control Barrier Function Lie derivatives match the ICLR 2025 paper.
   - The empirical observation noted in upstream line 1072—applying the CBF filter during the final denoising steps ($k \le 10$)—is implemented as our default `FAST_DEMO` and `BALANCED` operational mode.

3. **Hardware Adaptation Justification:**
   - Standalone kinematic simulation replaces legacy Linux MuJoCo 200 binaries.
   - Dual-solver architecture ensures both real-time analytical evaluation and general QP optimization on a 6GB VRAM NVIDIA RTX 4050 laptop GPU.
