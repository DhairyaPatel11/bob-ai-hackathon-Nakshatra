# ReadyLine Source Code (`src/`)

This directory contains the complete source code, signal processing pipelines, model training routines, and pre-trained model weights for the ReadyLine Predictive Maintenance Copilot.

---

## Directory Layout

```
src/
├── app.py                  # Streamlit flight-line readiness dashboard
├── pipeline.py             # Unified multi-modal readiness assessment orchestrator
├── train_model.py          # Turbofan engine XGBoost RUL model trainer
├── forecast.py             # IBM Granite TinyTimeMixer (TTM r2) zero-shot forecaster
├── vibration_features.py   # Bearing geometry kinematics and Hilbert envelope processing
├── bearing_model.py        # Gearbox vibration XGBoost RUL model trainer
├── explain.py              # Local IBM Granite 3.3 briefing generator (Ollama)
├── prioritize.py           # Supply-chain-aware priority queue ranker
├── etl_cmapss.py           # NASA C-MAPSS FD001 ingestion and Parquet builder
├── etl_ims.py              # NASA IMS Bearing dataset ingestion and Parquet builder
├── download_ims.py         # Automated NASA IMS dataset downloader (Kaggle mirror)
├── validate_combined.py    # End-to-end fleet validation runner (with TTM)
├── validate_fast.py        # Fast offline fleet validation runner
├── smoke_test_gearbox.py   # Vibration feature extraction verification test
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment configuration template
├── models/                 # Pre-trained models and feature schemas
│   ├── rul_model.json             # Engine XGBoost regressor
│   ├── feature_names.json         # Engine feature column list
│   ├── bearing_rul_model.json     # Gearbox XGBoost regressor
│   ├── bearing_feature_names.json # Gearbox feature column list
│   ├── predictions_val.csv        # Engine validation predictions
│   ├── predictions_test.csv       # Engine test predictions
│   └── bearing_predictions_val.csv# Gearbox validation predictions
└── output_real/            # Processed Parquet datasets
    ├── train_FD001_clean.parquet  # Cleaned C-MAPSS training rows
    ├── train_FD001_merged.parquet # C-MAPSS merged with service records
    ├── test_FD001_clean.parquet   # Cleaned C-MAPSS test rows
    ├── test_FD001_merged.parquet  # C-MAPSS test merged with service records
    ├── service_records.parquet    # Synthetic engine maintenance records
    ├── ims_bearing_features.parquet # Vibration condition indicators
    ├── ims_service_records.parquet# Synthetic gearbox maintenance records
    └── ims_bearing_merged.parquet # Bearing features merged with service records
```

---

## Core Scripts & Execution

### Launch the Application
```bash
streamlit run app.py
```

### Run Quick Verification
```bash
python validate_fast.py
```

### Retrain Models from Scratch
```bash
# Retrain engine model:
python train_model.py

# Retrain gearbox model:
python bearing_model.py
```
