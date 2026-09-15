# Setup and Installation Guide

This guide provides test-quality instructions to run ReadyLine locally from scratch.

---

## 1. System Prerequisites

- **Operating System:** Windows 10/11, macOS, or Ubuntu Linux
- **Python:** Python 3.11, 3.12, or 3.13
- **Git:** Installed and available on system PATH
- **Ollama (Optional for live LLM):** For local Granite briefings (download from [ollama.ai](https://ollama.ai))

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/DhairyaPatel11/bob-ai-hackathon-Nakshatra.git
cd bob-ai-hackathon-Nakshatra
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt):
.venv\Scripts\activate.bat

# Activate on Linux/macOS:
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r src/requirements.txt
```

### Step 4: Configure Environment Variables
```bash
cp src/.env.example src/.env
```
*(No edits are required by default. The project is pre-configured for local offline execution.)*

### Step 5: (Optional) Pull Local IBM Granite 3.3 via Ollama
If you wish to use live neural briefings rather than the built-in mock fallback:
```bash
# In a separate terminal:
ollama run granite3.3:2b
```

---

## 3. Running the Project

### Immediate Launch (Recommended)
ReadyLine includes pre-trained models in `src/models/` and pre-processed parquets in `src/output_real/`. You can launch the full dashboard immediately:

```bash
cd src
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Running Verification Tests
To run an automated test across the combined fleet (100 engines + 4 gearboxes):

```bash
# Fast-path verification (100% offline, completes in ~5 seconds):
python src/validate_fast.py

# Full-fidelity validation (including IBM Granite TTM neural forecasting):
python src/validate_combined.py
```

---

## 4. How to Verify It's Working

When `streamlit run src/app.py` loads in your browser, verify:
1. **Fleet Command Strip:** Displays 104 total assessed assets (100 aircraft engines, 4 drivetrain gearboxes).
2. **Readiness Summary:** Shows 4 mission-ready assets and 100 assets requiring maintenance.
3. **Maintenance Queue:** Shows a ranked list of grounded assets with modality badges (✈ Engine / ⚙ Gearbox), priority scores, and parts stock pills (`IN STOCK` or `PROCURE`).
4. **Diagnostic Briefing:** Clicking an asset card displays a concise 2-3 sentence briefing citing sensor values or bearing defect frequencies (BPFO/BPFI).
5. **Interactive Copilot:** Type *"Which assets are mission critical?"* in the chat box to test the natural language triage assistant.

---

## 5. Troubleshooting Table

| Symptom / Error | Root Cause | Solution |
|---|---|---|
| `ConnectionRefusedError: [WinError 10061]` on port 11434 | Ollama is not running on localhost. | **Automatic:** ReadyLine's built-in fast-fail cache detects offline Ollama and instantly falls back to deterministic structured briefings with zero delay. To use live LLM, run `ollama serve`. |
| `KeyError: sensor_N_prediction` or TTM Context Length Error | TTM model revision context size mismatch. | Ensure `forecast.py` uses revision `52-16-ft-l1-r2.1`. Do not use default 512-cycle revisions which exceed asset lifespan. |
| `WSCPA0000E: watsonx.ai authentication failed` | IBM Cloud trial account provisioning restriction. | ReadyLine is configured to use local IBM Granite via Ollama by default, bypassing cloud auth entirely. |
| Missing `output_real/` Parquet files | Repository cloned without data artifacts. | Run `python src/etl_cmapss.py` and `python src/etl_ims.py` to regenerate all Parquet tables from raw data. |
| `ModuleNotFoundError: No module named 'tsfm_public'` | `granite-tsfm` package not installed. | Run `pip install granite-tsfm`. Note: If absent, `forecast.py` gracefully falls back to rolling-window features without crashing. |
