
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

X = np.load(DATA_DIR / "sequences.npy")
y = np.load(DATA_DIR / "labels.npy", allow_pickle=True).astype(str)

# Summarize each video using position, variation, and movement.
mean = X.mean(axis=1)
std = X.std(axis=1)
movement = X[:, -1, :] - X[:, 0, :]
features = np.concatenate([mean, std, movement], axis=1)

models = {
    "SVM-RBF": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced"),
    ),
    "SVM-linear": make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", class_weight="balanced"),
    ),
    "KNN-3": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ),
}

cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=3, random_state=42
)

print("Repeated 5-fold cross-validation (15 scores per model)")
print("These are preliminary estimates from only 56 videos.\n")

results = []
for name, model in models.items():
    scores = cross_val_score(model, features, y, cv=cv, scoring="accuracy")
    results.append((scores.mean(), name, scores.std()))
    print(f"{name:<16} mean={scores.mean():.3f}  std={scores.std():.3f}")

print("\nBest average result:", max(results)[1])
