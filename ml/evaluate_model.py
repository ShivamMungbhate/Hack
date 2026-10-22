
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed_low_threshold"

X = np.load(DATA_DIR / "sequences.npy")
y = np.load(DATA_DIR / "labels.npy", allow_pickle=True).astype(str)

# Match the feature representation used by train_model.py and the API.
features = np.concatenate(
    [
        X.mean(axis=1),
        X.std(axis=1),
        X[:, -1, :] - X[:, 0, :],
    ],
    axis=1,
)

labels = np.unique(y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)

predictions = cross_val_predict(model, features, y, cv=cv)

print("5-fold out-of-fold classification report:")
print(classification_report(y, predictions, labels=labels, zero_division=0))

print("Confusion matrix (rows=true, columns=predicted):")
print("Label order:", list(labels))
print(confusion_matrix(y, predictions, labels=labels))

print(f"\nOut-of-fold accuracy: {np.mean(predictions == y):.3f}")
