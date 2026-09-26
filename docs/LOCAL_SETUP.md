# SAFEGEN 3D — Local Environment Setup Guide

**Operating System:** Windows 10/11 64-bit  
**Target Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM), 16GB RAM, or modern CPU  

---

## 1. Prerequisites

1. **Python:** 3.10, 3.11, or 3.12 (64-bit)
2. **Node.js:** v18.0.0 or higher with `npm`
3. **Git:** 2.30+
4. **CUDA (Optional for GPU acceleration):** CUDA 12.1 / 12.4 with compatible NVIDIA display drivers.

---

## 2. Step-by-Step Installation

### Step 1: Create Python Virtual Environment
Open PowerShell in the repository root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with CUDA Support
For NVIDIA GPUs with CUDA 12.4:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*(For CPU-only machines, simply run `pip install torch torchvision`)*

### Step 3: Install Core Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies
```powershell
cd app/frontend
npm install
cd ../..
```

---

## 3. Launching the System

### Start Frontend & Backend Concurrently
```powershell
.\scripts\start_dev.ps1
```
This script launches:
- **Backend API:** `http://127.0.0.1:8000` (FastAPI with Uvicorn and hot reload)
- **Frontend 3D UI:** `http://localhost:5173` (Vite dev server)

### Stopping Services
```powershell
.\scripts\stop_dev.ps1
```

---

## 4. Running Verification & Tests

### Automated Test Runner:
```powershell
.\scripts\run_tests.ps1
```

### Autonomous Verification Agent:
```powershell
.\scripts\run_verification.ps1
```
Generates full compliance reports at `verification/report.md`.
