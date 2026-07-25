import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_FILE = "data/scholarship_applications.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Number of records:", len(df))


# ==========================================
# 2. PREPARE THE DATA
# ==========================================

# Convert applied_date to datetime
df["applied_date"] = pd.to_datetime(df["applied_date"])

# Create numeric features from information available
# when the prediction is made.

# Month in which the application was submitted
df["applied_month"] = df["applied_date"].dt.month

# Day of the week
df["applied_dayofweek"] = df["applied_date"].dt.dayofweek

# Convert documents status into numbers
documents_mapping = {
    "Complete": 2,
    "Pending": 1,
    "Incomplete": 0
}

df["documents_score"] = df["documents_status"].map(documents_mapping)

# Convert application stage into numbers
stage_mapping = {
    "Document Collection": 0,
    "Verification": 1,
    "Sanctioned": 2,
    "Disbursed": 3
}

df["stage_score"] = df["stage"].map(stage_mapping)

# Convert scheme into numbers
scheme_mapping = {
    "State Merit Scholarship": 0,
    "Central Scholarship": 1,
    "SC Welfare Scholarship": 2
}

df["scheme_score"] = df["scheme"].map(scheme_mapping)


# ==========================================
# 3. SELECT FEATURES
# ==========================================

# IMPORTANT:
# We DO NOT use:
# - disbursed_date
# - outcome
#
# These are known only after the actual result.
# Therefore, they must not be used as prediction inputs.

features = [
    "applied_month",
    "applied_dayofweek",
    "documents_score",
    "stage_score",
    "scheme_score",
    "sanctioned_amount"
]

X = df[features]


# ==========================================
# 4. TARGET
# ==========================================

# Convert outcome into numbers
#
# On Time = 0
# Delayed = 1

df["target"] = df["outcome"].map({
    "On Time": 0,
    "Delayed": 1
})

y = df["target"]


# ==========================================
# 5. HANDLE MISSING VALUES
# ==========================================

X = X.fillna(0)
y = y.fillna(0)


# ==========================================
# 6. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. TRAIN MACHINE LEARNING MODEL
# ==========================================

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ==========================================
# 8. TEST MODEL
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ==========================================
# 9. EXAMPLE PREDICTION
# ==========================================

# Example: Predict risk for a new application

example_application = pd.DataFrame(
    [[
        7,      # Applied month: July
        0,      # Monday
        2,      # Documents Complete
        2,      # Sanctioned stage
        0,      # State Merit Scholarship
        50000   # Sanctioned amount
    ]],
    columns=features
)


# Get prediction probability
probabilities = model.predict_proba(example_application)[0]

predicted_class = model.predict(example_application)[0]

confidence = max(probabilities)


print("\nExample Prediction:")

# ==========================================
# 10. LOW-CONFIDENCE HANDLING
# ==========================================

if confidence < 0.60:

    print("Prediction Uncertain")
    print(
        "The model confidence is only",
        round(confidence * 100, 2),
        "%"
    )

else:

    if predicted_class == 1:
        print("⚠️ High Risk of Delay")
    else:
        print("✅ Low Risk of Delay")

    print(
        "Model Confidence:",
        round(confidence * 100, 2),
        "%"
    )