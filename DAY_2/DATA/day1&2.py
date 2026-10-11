


# ==============================================================
# SYSTEM AUTOMATION PROJECT: DAY 1 + DAY 2
# Day 1: Sensor input, rule evaluation, action and logging
# Day 2: Dataset loading, model training, model viewing, prediction
# ==============================================================

import random
import time
import sys
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==============================================================
# STEP 0 - SET PROJECT FILE PATHS AND CREATE DATA FOLDER
# ==============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "DATA"

DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "day2_sample.csv"
MODEL_PATH = DATA_DIR / "machine_state_model.pkl"
LOG_PATH = DATA_DIR / "day1_day2_log.txt"



# ==============================================================
# STEP 1 - CAPTURE ALL TERMINAL OUTPUT IN ONE LOG FILE
# ==============================================================

class Tee:
    """Display output in the terminal and save it to a file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


# ==============================================================
# DAY 1 - RULE-BASED SYSTEM AUTOMATION
# ==============================================================


# STEP 2 - TAKE SENSOR INPUT

def generate_sensor_data():
    """Generate sensor values in the same format as the original code."""

    return {
        "temperature": round(random.randint(50, 95)),
        "pressure": round(random.randint(1, 10), 2),
        "vibration": round(random.randint(0, 10), 2),
        "current": round(random.randint(2, 20), 2)
    }


# STEP 3 - DEFINE AUTOMATION RULES AND PRIORITIES

RULES = [
    {
        "name": "HIGH_TEMP",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 80,
        "action": "STOP_MACHINE"
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 7,
        "action": "INSPECT_MACHINE"
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 8,
        "action": "REDUCE_PRESSURE"
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 15,
        "action": "CHECK_MOTOR"
    }
]


# STEP 4 - EVALUATE RULES IN PRIORITY ORDER

def evaluate_rules(reading):
    sorted_rules = sorted(
        RULES,
        key=lambda rule: rule["priority"]
    )

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# STEP 5 - PERFORM THE ACTION

def take_action(action):
    """Return the response for every defined Day 1 action."""

    if action == "STOP_MACHINE":
        return "Machine stopped due to high temperature!"

    elif action == "INSPECT_MACHINE":
        return "High vibration detected. Inspect the machine!"

    elif action == "REDUCE_PRESSURE":
        return "High pressure detected. Reduce the pressure!"

    elif action == "CHECK_MOTOR":
        return "High current detected. Check the motor!"

    elif action == "CONTINUE_NORMAL":
        return "All parameters normal. Machine continues running."

    return f"No handler available for action: {action}"


# STEP 6 - RUN DAY 1 AUTOMATION AND RECORD EACH DECISION

def run_day1():
    print("\n" + "=" * 72)
    print("DAY 1 - RULE-BASED SYSTEM AUTOMATION")
    print("=" * 72)

    for tick in range(1, 11):

        # STEP 2: Read sensors
        reading = generate_sensor_data()

        # STEP 4: Select the first matching rule
        rule_name, action = evaluate_rules(reading)

        # STEP 5: Execute the simulated response
        outcome = take_action(action)

        print(f"\n[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}")

        # STEP 6: Log the decision in the combined log
        print(
            f"           LOG   -> Reading: {reading} | "
            f"Rule: {rule_name} | Action: {action}"
        )

        time.sleep(0.2)

    print("\nDay 1 automation completed: 10 cycles.")


# ==============================================================
# DAY 2 - MACHINE LEARNING
# ==============================================================


# STEP 7 - LOAD THE DATASET

def train_day2_model():

    print("\n" + "=" * 72)
    print("DAY 2 - MACHINE LEARNING")
    print("=" * 72)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {CSV_PATH}\n"
            "Please place day2_sample.csv inside the DATA folder."
        )

    df = pd.read_csv(CSV_PATH)

    print("\nSTEP 7 - LOAD DATASET")
    print(f"Dataset location: {CSV_PATH}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")

    print("\nFirst five dataset records:")
    print(df.head().to_string(index=False))

    # STEP 8 - SELECT INPUT FEATURES AND TARGET

    features = [
        "temperature",
        "pressure",
        "vibration",
        "current",
        "rpm"
    ]

    target = "state"

    required_columns = features + [target]
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing these required columns: {missing}"
        )

    # Check that required data is available and numeric.
    for column in features:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "The dataset contains missing values in required columns."
        )

    X = df[features]
    y = df[target]

    print("\nSTEP 8 - SELECT FEATURES AND MACHINE STATE")
    print(f"Input features: {features}")
    print(f"Target column: {target}")
    print("\nAvailable machine states:")
    print(y.value_counts().to_string())

    if y.nunique() < 2:
        raise ValueError(
            "The dataset must contain at least two machine-state classes."
        )

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each machine state needs at least two records for "
            "stratified training and testing."
        )

    # STEP 9 - SPLIT THE DATASET

    print("\nSTEP 9 - SPLIT TRAINING AND TESTING DATA")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    # STEP 10 - TRAIN RANDOM FOREST MODEL

    print("\nSTEP 10 - TRAIN RANDOM FOREST MODEL")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    print("Model training completed successfully.")

    # STEP 11 - EVALUATE MODEL

    print("\nSTEP 11 - EVALUATE MODEL")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )
    print(f"Class order: {list(model.classes_)}")

    # STEP 12 - SAVE THE TRAINED MODEL

    print("\nSTEP 12 - SAVE TRAINED MODEL")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully: {MODEL_PATH}")

    return model, features


# ==============================================================
# STEP 13 - VIEW THE TRAINED MODEL
# ==============================================================

def view_model(model, features):

    print("\n" + "=" * 72)
    print("STEP 13 - VIEW TRAINED MODEL")
    print("=" * 72)

    print(f"Model type: {type(model).__name__}")
    print(f"Number of trees: {len(model.estimators_)}")
    print(f"Input features: {features}")
    print(f"Predicted classes: {list(model.classes_)}")

    print("\nFeature importance:")

    importance_list = sorted(
        zip(features, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True
    )

    for feature, importance in importance_list:
        print(f"{feature}: {importance:.4f}")

    print("\nExample split rules from the first decision tree:")

    tree = model.estimators_[0].tree_

    for node in range(min(10, tree.node_count)):
        feature_index = tree.feature[node]

        if feature_index >= 0:
            print(
                f"Node {node}: "
                f"{features[feature_index]} <= "
                f"{tree.threshold[node]:.3f}"
            )
        else:
            print(f"Node {node}: leaf node")


# ==============================================================
# STEP 14 - LOAD SAVED MODEL AND PREDICT MACHINE STATE
# ==============================================================

def predict_machine_state(features):

    print("\n" + "=" * 72)
    print("STEP 14 - PREDICT MACHINE STATE")
    print("=" * 72)

    # Load the model saved in Step 12.
    model = joblib.load(MODEL_PATH)

    # Example input: edit these values to test a different reading.
    new_reading = {
        "temperature": 80.0,
        "pressure": 5.0,
        "vibration": 2.0,
        "current": 10.0,
        "rpm": 1700
    }

    input_data = pd.DataFrame(
        [new_reading],
        columns=features
    )

    predicted_state = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    print("\nNew machine sensor readings:")

    for sensor, value in new_reading.items():
        print(f"{sensor}: {value}")

    print(f"\nPredicted machine state: {predicted_state}")

    print("\nPrediction probabilities:")

    for state, probability in zip(model.classes_, probabilities):
        print(f"{state}: {probability * 100:.2f}%")


# ==============================================================
# STEP 15 - RUN THE COMPLETE DAY 1 + DAY 2 PROJECT
# ==============================================================

def main():

    # Create a fresh combined log for this complete run.
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:

        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)

        try:
            print("=" * 72)
            print("COMPLETE SYSTEM AUTOMATION PROJECT")
            print("DAY 1 + DAY 2")
            print("=" * 72)

            print(f"\nCombined log file: {LOG_PATH}")

            # DAY 1: Sensor input, rules, actions and logging
            run_day1()

            # DAY 2: Dataset, training, evaluation and saving
            model, features = train_day2_model()

            # DAY 2: Inspect the trained model
            view_model(model, features)

            # DAY 2: Load the saved model and make a prediction
            predict_machine_state(features)

            print("\n" + "=" * 72)
            print("ALL DAY 1 AND DAY 2 STEPS COMPLETED")
            print(f"Complete output saved to: {LOG_PATH}")
            print("=" * 72)

        finally:
            sys.stdout = original_stdout

    print(f"\nCombined log created: {LOG_PATH}")


if __name__ == "__main__":
    main()
