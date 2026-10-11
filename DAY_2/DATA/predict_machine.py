
import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
folder = Path(__file__).resolve().parent
model = joblib.load(folder / "machine_state_model.pkl")

# Enter new machine sensor readings
data = pd.DataFrame([{
    "temperature": 80.0,
    "pressure": 70.0,
    "vibration": 2.0,
    "current": 10.0,
    "rpm": 1700
}])

# Predict the machine condition
prediction = model.predict(data)[0]

print("Machine Sensor Readings:")
print(data.to_string(index=False))
print("\nPredicted Machine State:", prediction)

# Show estimated probabilities for each class
probabilities = model.predict_proba(data)[0]
print("\nPrediction probabilities:")

for state, probability in zip(model.classes_, probabilities):
    print(f"{state}: {probability * 100:.2f}%")
