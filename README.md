# Credit Risk Portfolio Analytics Dashboard

An interactive dashboard for exploring a credit portfolio, reviewing model performance, scoring accounts for risk review, identifying unusual accounts, and asking portfolio questions.

## Features
- Portfolio KPIs and segment-level default analysis
- Logistic Regression and XGBoost model comparison
- ROC-AUC, KS, Gini, and threshold analysis
- Account-level risk scoring with SHAP explanations
- Isolation Forest anomaly detection and human review queue
- Gemini-powered portfolio Q&A grounded in calculated results

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
