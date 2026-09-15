# Solution Overview: ReadyLine Predictive Maintenance Copilot

## 1. Core Mechanism: The P-F Interval Framing

Predictive maintenance in ReadyLine is structured around the **Potential Failure to Functional Failure (P-F) Interval**:

```
Asset Health
    ▲
100%│═══════════╗  (Normal Operation)
    │           ╚════╗
    │                ▼ [Point P: Detectable Potential Failure]
    │                 ╲
    │                  ╲  ◄── ReadyLine Operates Here (P-F Window)
    │                   ╲
    │                    ▼ [Point F: Functional Failure / Mission Loss]
  0%└─────────────────────────────────────────────► Time / Cycles
```

- **Point P (Potential Failure):** The earliest moment when physical degradation becomes detectable via telemetry (e.g., elevated high-pressure turbine temperature, rising vibration energy at outer-race ball-pass frequency).
- **Point F (Functional Failure):** The moment when the component fails in operation, causing mission abort, emergency diversion, or loss of asset.
- **P-F Interval:** The window between P and F. 

ReadyLine does not attempt to predict arbitrary health percentages. Instead, it continuously measures an asset's position along the P-F curve, calculates the remaining operational margin before Point F in flight cycles or operating hours, and triggers proactive maintenance when the predicted margin falls below the defined sortie mission envelope.

---

## 2. What Makes ReadyLine Different from Naive ML Approaches

| Dimension | Naive ML Approach | ReadyLine Predictive Copilot |
|---|---|---|
| **Fleet Modality** | Single sensor domain (e.g. only aircraft engines) | **Dual-modality mixed fleet:** Ingests turbofan thermodynamics + gearbox mechanical vibration |
| **Feature Signal** | Historical rolling stats only (backward-looking) | **Forward-looking foundation model:** IBM Granite TTM zero-shot neural forecasts + rolling stats |
| **Vibration Physics** | Generic RMS vibration thresholds | **Exact kinematics:** Computes BPFO, BPFI, BSF, FTF from physical bearing geometry |
| **Explainability** | Black-box feature importance bar charts | **Natural-language briefs:** Local IBM Granite 3.3 LLM explains *why* an asset is grounded |
| **Triage & Queue** | Sorted strictly by lowest RUL | **Supply-chain aware:** Factors RUL urgency, mission criticality, inspection dates, and parts lead time |
| **Deployment Model** | Cloud-dependent enterprise APIs | **DDIL / Edge-capable:** Fully local inference via Ollama and on-premise XGBoost/Parquet |

---

## 3. Multi-Sensor Corroboration & Physics-Grounded Signals

A primary cause of maintainer distrust in predictive algorithms is "false alarms" caused by single-sensor drift or noise. ReadyLine enforces multi-sensor corroboration across both modalities:

### Modality 1 — Aircraft Turbofan Engines (NASA C-MAPSS)
- Tracks 21 continuous thermodynamic and mechanical channels.
- Filters out 10 near-zero variance sensors (Sensors 1, 5, 6, 8, 10, 13, 15, 16, 18, 19).
- RUL predictions rely on corroborated trends across High-Pressure Turbine (HPT) outlet temperature (Sensor 4), Low-Pressure Turbine (LPT) outlet temperature (Sensor 3), and High-Pressure Compressor (HPC) static pressure (Sensor 11).
- IBM Granite TinyTimeMixer generates a 16-step forward forecast to evaluate degradation velocity (slope) before degradation accelerates exponentially.

### Modality 2 — Drivetrain Gearbox Bearings (NASA IMS Bearing Dataset)
- Utilizes 4-channel high-frequency accelerometry (20 kHz sampling rate) from Rexnord ZA-2115 double-row bearings run continuously to failure.
- Computes theoretical defect impact frequencies from manufacturer geometry:
  - **BPFO (Ball Pass Frequency Outer Race):** ~242.8 Hz @ 2000 RPM (outer race spalling)
  - **BPFI (Ball Pass Frequency Inner Race):** ~296.9 Hz @ 2000 RPM (inner race cracking)
  - **BSF (Ball Spin Frequency):** ~159.9 Hz @ 2000 RPM (roller surface defect)
  - **FTF (Fundamental Train Frequency):** ~7.6 Hz @ 2000 RPM (cage instability)
- Applies Hilbert transform envelope demodulation to extract spectral band energy around these exact kinematics, isolating bearing surface spalling from background gear mesh noise.

---

## 4. Key Design Decisions

### 1. XGBoost + IBM Granite TTM vs. Pure Deep Learning
End-to-end deep learning (LSTM, Transformer) on raw tabular degradation data is prone to overfitting and offers poor explainability for flight line operators. We adopted a hybrid architecture:
- **IBM Granite Time Series (TTM):** Operates zero-shot on sensor context windows to generate multi-channel forecast statistics (`fmean`, `fstd`, `fslope`).
- **Gradient Boosted Decision Trees (XGBoost):** Ingests raw readings, rolling window dynamics, and TTM forecast features. This yields sub-second inference, high resistance to outliers, and verifiable feature gains.

### 2. The TTM 52-16 Branch vs. Default 512-96
The standard pretrained TTM model assumes a 512-timestep context window. In military aerospace run-to-failure datasets like C-MAPSS FD001, turbofan engines fail between 128 and 362 cycles. A 512-cycle context window is mathematically impossible to satisfy. We identified, benchmarked, and integrated the `52-16-ft-l1-r2.1` checkpoint (52-cycle context, 16-step forecast), satisfying the real operational lifecycle constraints of the fleet.

### 3. Local IBM Granite 3.3 LLM (Ollama) vs. Cloud watsonx.ai
While ReadyLine includes a clean cloud integration adapter for IBM watsonx.ai, our active operational path uses local IBM Granite 3.3 2B served via Ollama:
- **Operational Reality:** Flight line maintenance crews operate in Denied, Disrupted, Intermittent, and Limited bandwidth (DDIL) environments where cloud connectivity is unavailable.
- **Resilience:** Bypasses external API authentication barriers (such as IBM Cloud WSCPA0000E account provisioning errors).
- **Zero-Latency Fallback:** An in-memory reachability cache checks Ollama once per session; if offline, it immediately falls back to structured deterministic briefings with zero latency.

### 4. Parts-Aware Prioritization Formula
ReadyLine ranks non-ready assets using an explicit, tunable composite formula:

Priority Score = 0.35 * Urgency + 0.25 * Criticality + 0.15 * TimePressure + 0.15 * Discrepancies + 0.10 * PartAvailability

Where:
- Urgency = 1 - min(RUL, 125) / 125
- Criticality = 1.0 if mission critical else 0.0
- Time Pressure = 1 - clamp(days to maint, 0, 90) / 90
- Discrepancy Load = min(discrepancies, 5) / 5
- Part Availability = 1.0 if part in stock else 1 - min(lead time, 30) / 30

---

## 5. User Experience: The Fleet Readiness Dashboard

The Streamlit copilot (`src/app.py`) provides maintainers with four functional layers:

1. **Fleet Readiness Command Strip:** High-level operational KPIs displaying total fleet count, mission-ready assets (GO), non-ready assets (NO-GO / MARGINAL), and mission-critical assets at risk.
2. **Priority Maintenance Queue:** Real-time ranked queue showing asset ID, modality badge (✈ Engine / ⚙ Gearbox), priority score bar, remaining RUL, open discrepancies, and spare part availability status (IN STOCK vs. PROCURE X days).
3. **Deep-Dive Diagnostic Card:** Selecting any asset opens a full briefing citing specific physical evidence (e.g. "Sensor 4 reading 1425.94 with 3 open write-ups since 2026-07-10" or "Elevated vibration energy at outer race (BPFO) defect frequency (3.25e-03) with cage instability").
4. **Natural-Language Ask ReadyLine Copilot:** Operators can query the fleet in plain English (e.g., "Which mission-critical assets have parts out of stock?" or "Why is Asset 34 grounded?") and receive contextual answers.
