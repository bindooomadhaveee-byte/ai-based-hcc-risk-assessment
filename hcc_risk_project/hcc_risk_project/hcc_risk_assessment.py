"""
AI-Based HCC (Hepatocellular Carcinoma) Risk Assessment
-------------------------------------------------------
A simple base project: loads (or synthesizes) patient data, trains ML models,
evaluates them, and predicts HCC risk for a new patient.

EDUCATIONAL USE ONLY - not a medical device, not for clinical decisions.

Usage:
    python hcc_risk_assessment.py train                  # synthetic data
    python hcc_risk_assessment.py train --csv data.csv   # your own data (needs 'hcc' column: 0/1)
    python hcc_risk_assessment.py predict                # interactive prediction
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "hcc_model.joblib"
SEED = 42

FEATURES = [
    "age", "sex_male", "cirrhosis", "hepatitis_b", "hepatitis_c",
    "alcohol_use", "diabetes", "bmi", "family_history",
    "afp", "alt", "ast", "platelets", "bilirubin", "albumin",
]


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def generate_synthetic_data(n=3000, seed=SEED):
    """Create a synthetic dataset loosely based on known HCC risk factors."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(25, 85, n),
        "sex_male": rng.binomial(1, 0.55, n),
        "cirrhosis": rng.binomial(1, 0.15, n),
        "hepatitis_b": rng.binomial(1, 0.10, n),
        "hepatitis_c": rng.binomial(1, 0.08, n),
        "alcohol_use": rng.binomial(1, 0.25, n),
        "diabetes": rng.binomial(1, 0.20, n),
        "bmi": np.round(rng.normal(26, 4.5, n), 1).clip(15, 50),
        "family_history": rng.binomial(1, 0.06, n),
        "afp": np.round(rng.lognormal(1.5, 1.2, n), 1),        # ng/mL
        "alt": np.round(rng.lognormal(3.4, 0.5, n), 0),        # U/L
        "ast": np.round(rng.lognormal(3.4, 0.5, n), 0),        # U/L
        "platelets": np.round(rng.normal(230, 70, n), 0).clip(30, 500),  # 10^9/L
        "bilirubin": np.round(rng.lognormal(-0.2, 0.5, n), 2), # mg/dL
        "albumin": np.round(rng.normal(4.0, 0.5, n), 2).clip(1.5, 5.5),  # g/dL
    })

    # Make cirrhosis-related labs more plausible
    c = df["cirrhosis"] == 1
    df.loc[c, "platelets"] = (df.loc[c, "platelets"] * 0.6).round(0)
    df.loc[c, "albumin"] = (df.loc[c, "albumin"] - 0.6).clip(1.5, 5.5).round(2)
    df.loc[c, "bilirubin"] = (df.loc[c, "bilirubin"] * 1.8).round(2)

    # Hidden risk function -> label
    logit = (
        -8.0
        + 0.04 * (df["age"] - 50)
        + 0.5 * df["sex_male"]
        + 2.2 * df["cirrhosis"]
        + 1.2 * df["hepatitis_b"]
        + 1.3 * df["hepatitis_c"]
        + 0.5 * df["alcohol_use"]
        + 0.4 * df["diabetes"]
        + 0.02 * (df["bmi"] - 25)
        + 0.7 * df["family_history"]
        + 0.9 * np.log1p(df["afp"])
        + 0.3 * np.log1p(df["bilirubin"])
        - 0.004 * (df["platelets"] - 200)
        - 0.4 * (df["albumin"] - 4.0)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["hcc"] = rng.binomial(1, prob)
    return df


def load_data(csv_path=None):
    if csv_path:
        df = pd.read_csv(csv_path)
        missing = [c for c in FEATURES + ["hcc"] if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        return df[FEATURES + ["hcc"]].dropna()
    return generate_synthetic_data()


# ----------------------------------------------------------------------------
# 2. Training & evaluation
# ----------------------------------------------------------------------------
def train(csv_path=None):
    df = load_data(csv_path)
    print(f"Dataset: {len(df)} patients | HCC prevalence: {df['hcc'].mean():.1%}")

    X, y = df[FEATURES], df["hcc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=SEED, n_jobs=-1),
    }

    best_name, best_model, best_auc = None, None, -1
    plt.figure(figsize=(6, 5))
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        print(f"\n=== {name} === AUC: {auc:.3f}")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=["No HCC", "HCC"]))
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        if auc > best_auc:
            best_name, best_model, best_auc = name, model, auc

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - HCC Risk Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix of the best model
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                          display_labels=["No HCC", "HCC"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    else:
        imp = np.abs(best_model.named_steps["clf"].coef_[0])
    order = np.argsort(imp)
    plt.figure(figsize=(7, 6))
    plt.barh(np.array(FEATURES)[order], imp[order])
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    joblib.dump({"model": best_model, "features": FEATURES, "name": best_name},
                MODEL_PATH)
    print(f"\nBest model: {best_name} (AUC {best_auc:.3f}) saved to {MODEL_PATH}")
    print("Saved plots: roc_curve.png, confusion_matrix.png, feature_importance.png")


# ----------------------------------------------------------------------------
# 3. Prediction
# ----------------------------------------------------------------------------
def risk_category(p):
    if p < 0.10:
        return "LOW"
    if p < 0.30:
        return "MODERATE"
    return "HIGH"


def predict_patient(patient: dict):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run: python hcc_risk_assessment.py train")
    bundle = joblib.load(MODEL_PATH)
    X = pd.DataFrame([patient])[bundle["features"]]
    p = float(bundle["model"].predict_proba(X)[0, 1])
    return p, risk_category(p)


PROMPTS = [
    ("age", "Age (years)", float),
    ("sex_male", "Male? (1=yes, 0=no)", int),
    ("cirrhosis", "Cirrhosis? (1/0)", int),
    ("hepatitis_b", "Hepatitis B? (1/0)", int),
    ("hepatitis_c", "Hepatitis C? (1/0)", int),
    ("alcohol_use", "Heavy alcohol use? (1/0)", int),
    ("diabetes", "Diabetes? (1/0)", int),
    ("bmi", "BMI", float),
    ("family_history", "Family history of liver cancer? (1/0)", int),
    ("afp", "AFP (ng/mL)", float),
    ("alt", "ALT (U/L)", float),
    ("ast", "AST (U/L)", float),
    ("platelets", "Platelets (10^9/L)", float),
    ("bilirubin", "Bilirubin (mg/dL)", float),
    ("albumin", "Albumin (g/dL)", float),
]


def interactive_predict():
    print("Enter patient details:\n")
    patient = {}
    for key, label, cast in PROMPTS:
        while True:
            try:
                patient[key] = cast(input(f"  {label}: "))
                break
            except ValueError:
                print("  Invalid input, try again.")
    p, cat = predict_patient(patient)
    print(f"\nEstimated HCC risk: {p:.1%}  ->  {cat} RISK")
    print("Note: educational tool only; consult a clinician for real assessment.")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI-Based HCC Risk Assessment")
    ap.add_argument("mode", choices=["train", "predict", "demo"])
    ap.add_argument("--csv", help="Optional CSV with feature columns + 'hcc' label")
    args = ap.parse_args()

    if args.mode == "train":
        train(args.csv)
    elif args.mode == "predict":
        interactive_predict()
    else:  # demo: one high-risk and one low-risk example
        high = dict(age=65, sex_male=1, cirrhosis=1, hepatitis_b=1, hepatitis_c=0,
                    alcohol_use=1, diabetes=1, bmi=29, family_history=1, afp=120,
                    alt=85, ast=110, platelets=90, bilirubin=2.1, albumin=3.0)
        low = dict(age=35, sex_male=0, cirrhosis=0, hepatitis_b=0, hepatitis_c=0,
                   alcohol_use=0, diabetes=0, bmi=22, family_history=0, afp=3,
                   alt=22, ast=20, platelets=260, bilirubin=0.7, albumin=4.4)
        for name, pt in [("High-risk example", high), ("Low-risk example", low)]:
            p, cat = predict_patient(pt)
            print(f"{name}: {p:.1%} -> {cat}")
