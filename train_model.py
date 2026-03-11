import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("video_proctoring_data.csv")

print("Dataset Loaded Successfully!")
print("Total Samples:", len(data))

# -----------------------------
# Separate Features and Labels
# -----------------------------
X = data.drop("label", axis=1)
y = data["label"]

# -----------------------------
# Encode Labels (Normal=0, Suspicious=1)
# -----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Label Mapping:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")

# -----------------------------
# Train-Test Split (80-20)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# -----------------------------
# Initialize Random Forest
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    class_weight="balanced"
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# Plot Confusion Matrix
# -----------------------------
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns

plt.figure()
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "proctoring_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel Saved Successfully as proctoring_model.pkl")


