# AI Risk Manager — Fraud Detection System

A production-grade fraud detector built for the Razorpay Buildathon (Track 02), featuring:
- LightGBM-based fraud classification with 92%+ recall
- SHAP explainability for every decision
- Audit trail logging for compliance
- Human-in-the-loop feedback system
- Guarded retraining pipeline with validation gate

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the demo

```bash
# (Optional) Reset and repopulate the database
python src/reset_db.py
python batch_simulate.py

# Launch the Streamlit dashboard
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Architecture

- `src/decision_pipeline.py` — main inference pipeline
- `src/explain.py` — SHAP-based risk explanation
- `src/audit_log.py` — SQLite audit trail
- `src/retrain_pipeline.py` — guarded retraining with validation gate
- `app.py` — Streamlit UI with 5 tabs

## Key files

- `models/base_model.pkl` — trained LightGBM classifier
- `data/paysim_*.pkl` — preprocessed PaySim dataset features
- `Fraud_Detection_Evaluation_Report.txt` — full evaluation on 552K test transactions

## Evaluation metrics

- Recall: 92.6% (fraud detection rate)
- Precision: 24.7% (at optimal threshold)
- Missed fraud cost: $113M (on held-out test set)
- False block cost: $75K (customer friction)
