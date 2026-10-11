

import joblib
from pathlib import Path

print("Loading machine state model...")

model_path = Path(__file__).resolve().parent / "machine_state_model.pkl"
model = joblib.load(model_path)

print("\nMODEL LOADED SUCCESSFULLY!")
print("Model type:", type(model).__name__)
print("Number of trees:", len(model.estimators_))
print("Machine states:", model.classes_)
print("Sensor inputs:", model.feature_names_in_)

print("\nModel file location:")
print(model_path)
