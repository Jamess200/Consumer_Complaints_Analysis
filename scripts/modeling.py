import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pathlib import Path

# === FILE PATHS ===
file_path = 'complaints_cleaned.csv'
figures_path = Path("figures/model_outputs")
figures_path.mkdir(parents=True, exist_ok=True)

# === SAVE PLOT FUNCTION ===
def save_plot(fig, filename):
    fig.savefig(figures_path / filename, dpi=300)
    plt.close(fig)

# === LOAD DATA IN CHUNKS ===
chunk_size = 500_000
chunks = []

print(" Estimating number of chunks...")
total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size))

print(" Loading cleaned data in chunks...")
for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), total=total_chunks, desc="Reading Chunks"):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
print(" Data loaded.")

# === CLEAN TARGET COLUMN ===
df = df.dropna(subset=['consumer_disputed?'])
df = df[df['consumer_disputed?'].isin(['Yes', 'No'])]
df['consumer_disputed_binary'] = df['consumer_disputed?'].map({'Yes': 1, 'No': 0})

# === FEATURES ===
features = [
    'product', 'sub_product', 'issue', 'submitted_via',
    'company_response_to_consumer', 'timely_response?'
]
df = df.dropna(subset=features)

# === ENCODING ===
label_encoders = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# === SPLIT ORIGINAL DATA ===
X_orig = df[features]
y_orig = df['consumer_disputed_binary']
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y_orig, stratify=y_orig, test_size=0.2, random_state=42
)

# === LOGISTIC REGRESSION BEFORE SMOTE ===
print("\n BEFORE Upsampling — Logistic Regression")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_orig, y_train_orig)
preds_orig = log_reg.predict(X_test_orig)

print("\n Logistic Regression (Before Upsampling):")
print(confusion_matrix(y_test_orig, preds_orig))
print(classification_report(y_test_orig, preds_orig))
print(f" ROC-AUC: {roc_auc_score(y_test_orig, preds_orig):.4f}")

# Save plots
fig = plt.figure()
RocCurveDisplay.from_estimator(log_reg, X_test_orig, y_test_orig)
plt.title("Logistic Regression ROC (Before Upsampling)")
save_plot(fig, "logreg_roc_before.png")

fig = plt.figure()
PrecisionRecallDisplay.from_estimator(log_reg, X_test_orig, y_test_orig)
plt.title("Logistic Regression PR (Before Upsampling)")
save_plot(fig, "logreg_pr_before.png")

# === UPSAMPLING WITH SMOTE ===
print("\n Upsampling with SMOTE...")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X_orig, y_orig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === LOGISTIC REGRESSION AFTER SMOTE ===
print("\n AFTER Upsampling — Logistic Regression")
log_reg.fit(X_train, y_train)
preds_balanced = log_reg.predict(X_test)

print("\n Logistic Regression (After Upsampling):")
print(confusion_matrix(y_test, preds_balanced))
print(classification_report(y_test, preds_balanced))
print(f" ROC-AUC: {roc_auc_score(y_test, preds_balanced):.4f}")

# Save plots
fig = plt.figure()
RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression ROC (After SMOTE)")
save_plot(fig, "logreg_roc_after.png")

fig = plt.figure()
PrecisionRecallDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression PR (After SMOTE)")
save_plot(fig, "logreg_pr_after.png")

# === XGBOOST AFTER SMOTE ===
print("\n AFTER Upsampling — XGBoost")
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)

print("\n XGBoost (After Upsampling):")
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))
print(f" ROC-AUC: {roc_auc_score(y_test, xgb_preds):.4f}")

# Save plots
fig = plt.figure()
RocCurveDisplay.from_estimator(xgb_clf, X_test, y_test)
plt.title("XGBoost ROC (After SMOTE)")
save_plot(fig, "xgb_roc_after.png")

fig = plt.figure()
PrecisionRecallDisplay.from_estimator(xgb_clf, X_test, y_test)
plt.title("XGBoost PR (After SMOTE)")
save_plot(fig, "xgb_pr_after.png")

fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_clf, importance_type='gain', ax=ax)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
save_plot(fig, "xgb_feature_importance.png")

input("\n Press ENTER to close the script...")
