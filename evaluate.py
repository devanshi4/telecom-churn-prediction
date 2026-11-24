"""Evaluate the churn model on the held-out test set and write the results used in the README.

Reports more than accuracy: only about a quarter of customers churn, so a model that
always says "stays" would already be ~73% accurate.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from train_model import build_model, load_and_prepare, split

OUT = Path("docs")


def main():
    X, y, _ = load_and_prepare()
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_model().fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "test_customers": int(len(y_test)),
        "churn_rate": round(float(y.mean()), 3),
        "baseline_accuracy_always_stays": round(float(1 - y_test.mean()), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp),
                             "false_negative": int(fn), "true_positive": int(tp)},
    }

    # Global explanation: mean |SHAP| per feature across the test set.
    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    importance = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(X_test.columns, importance), key=lambda t: -t[1])
    metrics["top_drivers_mean_abs_shap"] = [[name, round(float(v), 3)] for name, v in ranked[:8]]

    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    shap.summary_plot(shap_values, X_test, max_display=10, show=False, plot_size=(8, 5))
    plt.title("What drives churn (SHAP, test set)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
