'''
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

# Load extracted hand-landmark sequences and labels.
X = np.load(DATA_DIR / "sequences.npy")
y = np.load(DATA_DIR / "labels.npy", allow_pickle=True).astype(str)

# Baseline: summarize each 30-frame sequence with mean and
# standard deviation for every landmark feature.
X_mean = X.mean(axis=1)
X_std = X.std(axis=1)
X_features = np.concatenate([X_mean, X_std], axis=1)

# Keep a held-out test set. Stratification preserves class balance.
X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced"),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Save the trained model.
import joblib

joblib.dump(model, MODEL_DIR / "signbridge_svm.joblib")

with open(MODEL_DIR / "sign_labels.json", "w", encoding="utf-8") as f:
    json.dump(sorted(np.unique(y).tolist()), f, indent=2)

print(f"\nModel saved to: {MODEL_DIR / 'signbridge_svm.joblib'}")
'''


import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

X = np.load(DATA_DIR / "sequences.npy")
y = np.load(DATA_DIR / "labels.npy", allow_pickle=True).astype(str)

# Match the features used in compare_models.py.
mean = X.mean(axis=1)
std = X.std(axis=1)
movement = X[:, -1, :] - X[:, 0, :]
features = np.concatenate([mean, std, movement], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_DIR / "signbridge_svm.joblib")

with (MODEL_DIR / "sign_labels.json").open("w", encoding="utf-8") as f:
    json.dump(sorted(np.unique(y).tolist()), f, indent=2)

print(f"\nModel saved to: {MODEL_DIR / 'signbridge_svm.joblib'}")
