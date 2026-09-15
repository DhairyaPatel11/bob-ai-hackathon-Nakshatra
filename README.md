# ReadyLine - Mission Readiness & Predictive Maintenance Copilot

> AI-powered predictive maintenance and mission readiness copilot for mixed military fleets, combining aerodynamic sensor telemetry, gearbox vibration diagnostics, IBM Granite Time Series (TTM) zero-shot forecasting, and local LLM briefings.

---

## Team

| Field | Value |
|---|---|
| **Team Name** | Nakshatra |
| **Track** | AI |
| **Team Lead** | Dhairya Patel - 24ec096@charusat.edu.in |
| **Members** | Dhairya Patel (Lead) |

---

## Problem Statement

Military aircraft, combat vehicles, and rotorcraft suffer from depressed mission-capable rates because maintenance operates on rigid calendar schedules rather than actual component degradation, while gigabytes of onboard Health and Usage Monitoring System (HUMS) telemetry go unanalyzed. This failure mode drives $90 billion annually in Department of Defense sustainment expenditure and persistent readiness crises (such as the F-35 fleet's mission-capable rate declining from 67% in FY2021 to 44% in FY2025). Despite 20 years of the DoD Condition-Based Maintenance Plus policy (DoDI 4151.22), GAO report GAO-23-105556 found implementation remains fundamentally stalled due to fragmented tools and lack of actionable flight-line decision support.

---

## Solution

ReadyLine delivers an end-to-end predictive maintenance copilot that ingests real-time aerodynamic and mechanical degradation feeds across two distinct asset classes (turbofan engines and drivetrain gearboxes) into a unified operational schema. The system pairs an XGBoost regression backbone with IBM Granite Time Series (TinyTimeMixer / TTM) foundation model zero-shot forecasting to predict Remaining Useful Life (RUL), synthesizes physical condition indicators into concise natural-language briefings via local IBM Granite 3.3 (Ollama), and automatically sequences grounded assets in a supply-chain-aware priority queue factoring parts lead times and mission criticality.

---

## Key Features

- **Dual-Modality Mixed-Fleet Assessment:** Simultaneously tracks aero-thermodynamic degradation (NASA C-MAPSS turbofans) and high-frequency mechanical vibration (NASA IMS Rexnord ZA-2115 bearings) in one normalized operational queue.
- **IBM Granite Time Series (TTM) Forecasting:** Deploys IBM Granite TinyTimeMixer (branch 52-16-ft-l1-r2.1) to generate zero-shot multi-step sensor forecasts, capturing degradation velocity alongside historical rolling trends.
- **Physics-Grounded Granite Briefings:** Local IBM Granite 3.3 LLM translates complex multi-sensor anomalies into 2-3 sentence maintenance officer briefings citing exact defect frequencies (BPFO, BPFI) and wear indicators.
- **Supply-Chain & Parts-Aware Prioritization:** Ranks non-ready assets using a composite multi-factor formula integrating RUL failure urgency (40%), mission-critical status (25%), scheduled maintenance deadlines (20%), open discrepancies (15%), and depot spare-parts lead times.
- **Flight-Line Streamlit Copilot:** Interactive web interface providing fleet-wide readiness KPIs, triage queues, drill-down sensor time series, and an operator Q&A assistant.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.11+ (tested on Python 3.13) |
| **Frameworks** | Streamlit, XGBoost, scikit-learn, SciPy, Pandas, NumPy, PyArrow |
| **IBM Technologies** | IBM Bob (AI pair programmer used throughout development), IBM Granite Time Series (TinyTimeMixer / TTM r2), IBM Granite 3.3 2B (Ollama local inference), watsonx.ai SDK architecture |
| **Databases** | Apache Parquet columnar storage (fast zero-copy dataset caching) |
| **Other** | NASA C-MAPSS FD001 Turbofan Dataset, NASA IMS Bearing Vibration Dataset, Plotly |

*Note on IBM Bob:* IBM Bob was used as our primary development partner throughout the hackathon lifecycle to architect pipelines, implement Hilbert-transform vibration signal processing, debug TTM context length mismatches, and integrate dual-modality schemas.

---

## Repository Structure

```
├── submission.yaml         # Hackathon submission metadata
├── README.md               # Project overview and quickstart
├── docs/                   # Detailed documentation
│   ├── problem-statement.md# Mission readiness crisis and DoD CBM+ context
│   ├── solution-overview.md# P-F interval mechanics, design decisions, and UX
│   ├── architecture.md     # Mermaid data flow and system component breakdown
│   └── setup-guide.md      # Zero-context installation and run walkthrough
├── src/                    # Source code and pre-trained models
│   ├── app.py              # Streamlit fleet readiness dashboard
│   ├── pipeline.py         # Unified multi-modal readiness assessment orchestrator
│   ├── train_model.py      # Engine XGBoost RUL training and evaluation
│   ├── forecast.py         # IBM Granite TTM zero-shot time-series forecaster
│   ├── vibration_features.py# Bearing geometry and defect frequency signal processor
│   ├── bearing_model.py    # Gearbox vibration XGBoost RUL model
│   ├── explain.py          # Local IBM Granite briefing generator (Ollama)
│   ├── prioritize.py       # Supply-chain-aware priority ranking queue
│   ├── etl_cmapss.py       # C-MAPSS data ingestion and service record synthesis
│   ├── etl_ims.py          # NASA IMS vibration ETL and parquet builder
│   ├── download_ims.py     # Automated IMS dataset downloader
│   ├── validate_combined.py# Full validation runner (with TTM)
│   ├── validate_fast.py    # Fast validation runner (instant offline demo)
│   ├── smoke_test_gearbox.py# Vibration feature extraction verification test
│   ├── models/             # Pre-trained XGBoost models and feature JSON schemas
│   ├── output_real/        # Pre-processed parquet datasets (engine + gearbox)
│   ├── requirements.txt    # Python package dependencies
│   ├── .env.example        # Environment variable template
│   └── README.md           # Module-level documentation
├── demo/                   # Demonstration artifacts
│   ├── demo-video-link.txt # Hosted video demonstration link
│   ├── live-demo-url.txt   # Live deployment status
│   ├── screenshots/        # Application interface captures
│   └── README.md           # Demo catalog and guide
├── presentation/           # Pitch presentation deck
├── CONTRIBUTING.md          # Submission instructions
└── .gitignore              # Repository exclusions
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/DhairyaPatel11/bob-ai-hackathon-Nakshatra.git
cd bob-ai-hackathon-Nakshatra
```

### 2. Install dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r src/requirements.txt
```

### 3. (Optional) Run local Granite LLM via Ollama
```bash
# In a separate terminal:
ollama run granite3.3:2b
```
*Note: If Ollama is not running, ReadyLine automatically uses its built-in deterministic fallback briefing engine with zero delay.*

### 4. Launch the Streamlit Copilot Dashboard
```bash
cd src
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Demo

| Artifact | Link |
|---|---|
| Demo Video | demo/demo-video-link.txt |
| Live Demo | demo/live-demo-url.txt (Local flight-line deployment) |
| Screenshots | demo/screenshots/ |
| Presentation | presentation/ |

---

## Known Limitations

- **TTM Generalization vs. Truncated Sequences:** Integrating IBM Granite TTM forward-looking features reduced validation RMSE on full trajectories from 33.60 to 27.91 cycles. However, on test set FD001 (where sequences are cut off prematurely before failure), test RMSE was 53.15 cycles versus 46.58 cycles for rolling statistics alone, highlighting that zero-shot foundation models require continuous historical context.
- **Local Granite vs. Cloud watsonx.ai:** Due to IBM Cloud account provisioning restrictions (WSCPA0000E), we pivoted to running IBM Granite 3.3 locally via Ollama. While this resolved cloud access blocks, it also represents an intentional operational advantage for disconnected flight-line deployments (DDIL).
- **Synthetic Maintenance Records:** While engine thermodynamics (C-MAPSS) and gearbox accelerations (IMS) are 100% real run-to-failure NASA datasets, maintenance history and depot spare parts inventory are synthesized via statistical correlation with asset degradation tiers because operational DoD logistics data is classified.
- **Point Estimates:** RUL predictions currently output point estimates rather than full probabilistic confidence bounds.

---

## What We're Most Proud Of

1. **Solving the TTM Context-Length Architectural Barrier:** Default IBM Granite TTM checkpoints assume 512-cycle context windows. In C-MAPSS FD001, turbofans fail completely within 128 to 362 cycles, making the default model mathematically impossible to execute on real asset histories. We researched and successfully deployed the fine-tuned `52-16-ft-l1-r2.1` checkpoint (52-cycle context, 16-step forecast), proving real foundation model integration on military equipment lifecycles.
2. **True Dual-Modality Fusion:** Rather than presenting a single-sensor proof-of-concept, ReadyLine models two entirely different physical failure modes (turbofan aerodynamic degradation and gearbox bearing race spalling) under a single unified schema and queue.
3. **Supply-Chain Grounded Prioritization:** Connecting machine learning RUL directly with spare-parts depot lead time ensures maintenance commanders never prioritize an asset that cannot be repaired over one whose parts are sitting on the shelf.
