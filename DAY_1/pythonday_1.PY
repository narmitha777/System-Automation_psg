import random
import time

LOG_FILE = "day1_log.txt"


# STEP 1 & STEP 6 — Generate simulated sensor data
def generate_sensor_data():
    """Generate random sensor readings."""
    return {
        "temperature": round(random.randint(50, 95)),
        "pressure": round(random.randint(1, 10), 2),
        "vibration": round(random.randint(0, 10), 2),
        "current": round(random.randint(2, 20), 2)
    }


# STEP 2, STEP 7 & STEP 8 — Define automation rules
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


# STEP 3 — Evaluate rules
def evaluate_rules(reading, rules):
    """Evaluate rules in priority order."""
    sorted_rules = sorted(rules, key=lambda rule: rule["priority"])

    for rule in sorted_rules:
        if rule["condition"](reading):
            return rule["name"], rule["action"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"


# STEP 4 — Trigger an action
def take_action(action):
    """Return a message describing what happens for an action."""

    if action == "STOP_MACHINE":
        return "Machine stopped due to high temperature!"

    elif action == "REDUCE_PRESSURE":
        return "High pressure detected. Reduce the pressure!"

    elif action == "INSPECT_MACHINE":
        return "High vibration detected. Inspect the machine!"

    elif action == "CONTINUE_NORMAL":
        return "All parameters normal. Machine continues running."

    else:
        return f"No handler available for action: {action}"

# STEP 5 — Log every decision
def log_decision(reading, rule_name, action):
    """Append sensor readings and decisions to the log file."""
    with open(LOG_FILE, "a") as file:
        file.write(
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )
        


# MAIN PROGRAM
def main():
    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)

        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)

        time.sleep(0.2)


if __name__ == "__main__":
    main()