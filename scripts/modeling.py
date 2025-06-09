import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load cleaned CSV file
file_path = 'complaints_cleaned.csv'
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

# Clean target column
df = df.dropna(subset=['consumer_disputed?'])
df = df[df['consumer_disputed?'].isin(['Yes', 'No'])]
df['consumer_disputed_binary'] = df['consumer_disputed?'].map({'Yes': 1, 'No': 0})

# Features for model
features = [
    'product', 'sub_product', 'issue', 'submitted_via',
    'company_response_to_consumer', 'timely_response?'
]
df = df.dropna(subset=features)

# Encode features
label_encoders = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Prepare data
X_orig = df[features]
y_orig = df['consumer_disputed_binary']

# Split before balancing
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
    X_orig, y_orig, stratify=y_orig, test_size=0.2, random_state=42)

# Train Logistic Regression on original (imbalanced) data
print("\n BEFORE Upsampling — Logistic Regression")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_orig, y_train_orig)
preds_orig = log_reg.predict(X_test_orig)

print("\n Logistic Regression (Before Upsampling):")
print(confusion_matrix(y_test_orig, preds_orig))
print(classification_report(y_test_orig, preds_orig))

# ROC-AUC and Curves (before upsampling)
roc_auc_orig = roc_auc_score(y_test_orig, preds_orig)
print(f" Logistic Regression ROC-AUC (Before Upsampling): {roc_auc_orig:.4f}")

RocCurveDisplay.from_estimator(log_reg, X_test_orig, y_test_orig)
plt.title("Logistic Regression ROC Curve (Before Upsampling)")
plt.show()

PrecisionRecallDisplay.from_estimator(log_reg, X_test_orig, y_test_orig)
plt.title("Logistic Regression Precision-Recall Curve (Before Upsampling)")
plt.show()

# Handle imbalance using SMOTE
print("\n Upsampling with SMOTE...")
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X_orig, y_orig)

# Split after SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Train Logistic Regression on balanced data
print("\n AFTER Upsampling — Logistic Regression")
log_reg.fit(X_train, y_train)
preds_balanced = log_reg.predict(X_test)

print("\n Logistic Regression (After Upsampling):")
print(confusion_matrix(y_test, preds_balanced))
print(classification_report(y_test, preds_balanced))

# ROC-AUC and Curves (after upsampling)
roc_auc_balanced = roc_auc_score(y_test, preds_balanced)
print(f" Logistic Regression ROC-AUC (After Upsampling): {roc_auc_balanced:.4f}")

RocCurveDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression ROC Curve (After Upsampling)")
plt.show()

PrecisionRecallDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression Precision-Recall Curve (After Upsampling)")
plt.show()

# Train XGBoost on balanced data
print("\n AFTER Upsampling — XGBoost")
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
xgb_preds = xgb_clf.predict(X_test)

print("\n XGBoost (After Upsampling):")
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))

# ROC-AUC and Curves (XGBoost)
roc_auc_xgb = roc_auc_score(y_test, xgb_preds)
print(f" XGBoost ROC-AUC (After Upsampling): {roc_auc_xgb:.4f}")

RocCurveDisplay.from_estimator(xgb_clf, X_test, y_test)
plt.title("XGBoost ROC Curve (After Upsampling)")
plt.show()

PrecisionRecallDisplay.from_estimator(xgb_clf, X_test, y_test)
plt.title("XGBoost Precision-Recall Curve (After Upsampling)")
plt.show()

# Show feature importance
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xgb_clf, importance_type='gain', ax=ax)
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

input("\n Press ENTER to close the script...")
