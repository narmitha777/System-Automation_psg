
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Use the folder where this Python script is located
folder = Path(__file__).resolve().parent
csv_path = folder / "day2_sample.csv"
model_path = folder / "machine_state_model.pkl"

# Load dataset
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

features = [
    "temperature", "pressure", "vibration", "current", "rpm"
]
target = "state"

# Validate columns and clean values
required = features + [target]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target] = df[target].astype("string").str.strip().str.upper()
df = df.dropna(subset=required)

X = df[features]
y = df[target]

if y.nunique() < 2:
    raise ValueError("Training requires at least two different machine states.")

if y.value_counts().min() < 2:
    raise ValueError(
        "Each state needs at least two records for a stratified split."
    )

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Predict labels for the unseen test records
y_pred = model.predict(X_test)

# Calculate accuracy
correct = (y_pred == y_test).sum()
total = len(y_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n--- MODEL ACCURACY ---")
print(f"Correct predictions: {correct}/{total}")
print(f"Incorrect predictions: {total - correct}")
print(f"Test accuracy: {accuracy:.2f}%")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred, labels=model.classes_))
print("Class order:", model.classes_)


# Save the trained model
joblib.dump(model, model_path)
print(f"\nTraining complete! Model saved to:\n{model_path}")

