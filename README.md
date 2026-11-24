# Customer Churn Prediction

A machine learning app that predicts which telecom customers are likely to cancel, and explains why. An XGBoost model trained on 7,043 customers is served in a Streamlit dashboard, with SHAP showing which factors drive each prediction.

**Stack:** Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit

> Part of my portfolio at **[devanshi4.github.io](https://devanshi4.github.io)**

## Highlights

- **Who is about to leave:** ranks customers by churn risk with a ROC-AUC of 0.82 on held-out data.
- **And why:** SHAP shows the top factors behind every single prediction.
- **What-if in one click:** change a customer's contract or bill and watch the risk update.

---

## The problem

Keeping a customer is cheaper than winning a new one, but a retention team can only call so many people. They need to know who is at risk and why: a high monthly bill calls for a different offer than a month-to-month contract. A risk score with no reason attached is hard to act on.

## How it works

```mermaid
flowchart LR
    A[Telco churn CSV] --> B[train_model.py<br/>clean + encode]
    B --> C[XGBoost classifier]
    C --> D[(model + encoders<br/>.pkl files)]
    D --> E[Streamlit app]
    E --> F[Risk score]
    E --> G[SHAP explanation]
```

1. **Train.** `train_model.py` cleans the IBM Telco dataset, encodes the categorical columns, trains XGBoost on a stratified 80/20 split, and saves the model, the encoders and the column order. `evaluate.py` measures it on the held-out 20%.
2. **Score.** In the app, a user sets a customer profile (contract, tenure, services, charges) and gets a churn probability.
3. **Explain.** SHAP ranks the features that most influenced that customer's score.

## Design decisions

- **XGBoost for tabular data.** Gradient-boosted trees are a strong default for mixed numeric and categorical columns, and they work directly with SHAP's fast tree explainer.
- **Explanations, not just scores.** Each prediction comes with its top drivers, so the retention team knows what to offer.
- **Training and serving share the same encoders.** The encoders and column order are saved with the model and reused in the app, so a value like "Month-to-month" becomes the same number in both.
- **Scenario testing built in.** Changing one field in the sidebar (for example, a one-year contract) re-scores the customer immediately.

## Results

Measured on 1,409 held-out customers (a stratified 20% test split). Reproduce with `python evaluate.py`, which writes [`docs/metrics.json`](docs/metrics.json).

| Metric | Value | What it means |
|---|---|---|
| ROC-AUC | **0.82** | How well the model ranks risky customers above safe ones |
| Precision (churners) | **0.60** | Of customers flagged as at risk, 60% really left |
| Recall (churners) | **0.52** | The model catches about half of the customers who leave |
| Accuracy | **0.78** | Compared with 0.735 for a model that always predicts "stays" |

Only 26.5% of customers churn, so accuracy alone flatters any model. Precision and recall show the trade-off the retention team actually faces. Lowering the 0.5 decision threshold would catch more churners, at the cost of more false alarms.

![SHAP summary: contract type, tenure and monthly charges drive churn the most](docs/shap_summary.png)

**What drives churn:** contract type matters most, with month-to-month customers far more likely to leave. Then come tenure (new customers leave more) and monthly charges (higher bills raise the risk). Customers without online security or tech support are also at higher risk.

## Run it locally

You need Python 3.9+ and the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) saved as `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the repo folder.

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train and evaluate the model:

   ```bash
   python train_model.py
   ```

   ```bash
   python evaluate.py
   ```

3. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

## Project structure

```
├── train_model.py     # Clean, encode, train and save the model
├── evaluate.py        # Held-out metrics and SHAP chart -> docs/
├── app.py             # Streamlit dashboard with SHAP explanations
└── requirements.txt
```

## Status and next steps

- Tune the decision threshold for the retention team's budget, not the default 0.5.
- Replace label encoding with one-hot encoding for columns that have no natural order.
- Deploy the app to Streamlit Community Cloud.

---

Built by **Devanshi Pandya** · [Portfolio](https://devanshi4.github.io) · [LinkedIn](https://www.linkedin.com/in/devanshi4/)
