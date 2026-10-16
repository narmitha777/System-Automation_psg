"""
train_wear_model.py
-------------------
Trains a Random Forest classifier to predict joint wear states from historical sensor data.
Part of 23IF02 System Automation - Robotic Arm Joint Health Monitoring System.

Features monitored across 6-axis arm key joints:
- Shoulder: Torque (Nm), Temperature (°C), Vibration (mm/s)
- Elbow:    Torque (Nm), Temperature (°C), Vibration (mm/s)
- Wrist:    Torque (Nm), Temperature (°C), Vibration (mm/s)

Target classes:
- NORMAL
- MODERATE_WEAR
- SEVERE_WEAR
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "synthetic_robot_arm_data.csv"
MODEL_PATH = BASE_DIR / "wear_prediction_model.pkl"

FEATURES = [
    "shoulder_torque", "shoulder_temp", "shoulder_vibration",
    "elbow_torque", "elbow_temp", "elbow_vibration",
    "wrist_torque", "wrist_temp", "wrist_vibration"
]
TARGET = "wear_status"

def train_model():
    print("=" * 60)
    print("TRAINING PREDICTIVE JOINT WEAR MODEL")
    print("=" * 60)
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first.")
        
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {DATASET_PATH.name} ({len(df)} records, {len(df.columns)} columns)")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Train / test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Test Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print("Classes:", list(clf.classes_))
    
    print("\n--- FEATURE IMPORTANCES ---")
    for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<22}: {imp:.4f}")
        
    # Save model and metadata
    package = {
        "model": clf,
        "features": FEATURES,
        "classes": list(clf.classes_)
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n[OK] Model successfully saved to {MODEL_PATH.name}")

if __name__ == "__main__":
    train_model()
