import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ---------------------------
# 1. Load and Preprocess Dataset
# ---------------------------
filename = "training_dataset.csv"
df = pd.read_csv(filename)
print("Columns in dataset:", df.columns.tolist())

# The dataset has columns:
# [Algorithm Used, Memory Depth, Mutation Rate, Opponent_Name, Coop Rate, Evaluation]

# Define the list of non-numeric columns
non_numeric_cols = ["Algorithm Used", "Opponent_Name"]

# Create and store LabelEncoders for these columns
encoders = {}
for col in non_numeric_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Ensure the target "Evaluation" is numeric
df["Evaluation"] = df["Evaluation"].astype(int)

# Separate features (X) and target (y)
# Drop "Evaluation" to get the features
X = df.drop("Evaluation", axis=1)
y = df["Evaluation"]

# ---------------------------
# 2. Standardize Features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3. Train/Test Split and Model Training
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train four classifiers
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Evaluate models on the test set (optional)
print("\nModel Accuracies on Test Set:")
print("Logistic Regression:", accuracy_score(y_test, log_reg.predict(X_test)))
print("Decision Tree:", accuracy_score(y_test, tree.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, rf.predict(X_test)))
print("SVM:", accuracy_score(y_test, svm.predict(X_test)))
print("")

# ---------------------------
# 4. Define safe_transform function
# ---------------------------
def safe_transform(val, encoder):
    """
    Transforms a value using the given encoder.
    If the value is not seen during training, returns -1.
    """
    if val in encoder.classes_:
        return encoder.transform([val])[0]
    else:
        return -1

# ---------------------------
# 5. Create Multiple Sample Inputs and Predict
# ---------------------------
# We'll define sample inputs with the 5 columns:
# "Algorithm Used", "Memory Depth", "Mutation Rate", "Opponent_Name", "Coop Rate"
# We want to see if the model predicts them as good (1) or bad (0).

sample_inputs = [
    # Sample 1
    {
        "Algorithm Used": "Hill Climbing",
        "Memory Depth": 1,
        "Mutation Rate": 0.05,
        "Opponent_Name": "GTFT",
        "Coop Rate": 0.2
    },
    # Sample 2
    {
        "Algorithm Used": "Genetic Algorithm",
        "Memory Depth": 2,
        "Mutation Rate": 0.01,
        "Opponent_Name": "ShortTermTitForTat",
        "Coop Rate": 0.80
    },
    # Sample 3
    {
        "Algorithm Used": "Simulated Annealing",
        "Memory Depth": 3,
        "Mutation Rate": 0.05,
        "Opponent_Name": "AlwaysDefect",
        "Coop Rate": 0.50
    },
    # Sample 4
    {
        "Algorithm Used": "Tabular Seach",
        "Memory Depth": 1,
        "Mutation Rate": 0.1,
        "Opponent_Name": "RandomStrategy",
        "Coop Rate": 0.65
    },
    # Sample 5
    {
        "Algorithm Used": "Genetic Algorithm",
        "Memory Depth": 5,
        "Mutation Rate": 0.05,
        "Opponent_Name": "TitForTwoTats",
        "Coop Rate": 0.9
    },
    # Sample 6
    {
        "Algorithm Used": "Simulated Annealing",
        "Memory Depth": 4,
        "Mutation Rate": 0.01,
        "Opponent_Name": "HardJoss",
        "Coop Rate": 0.75
    },
    # Sample 7
    {
        "Algorithm Used": "Hill Climbing",
        "Memory Depth": 3,
        "Mutation Rate": 0.1,
        "Opponent_Name": "Grudger",
        "Coop Rate": 0.40
    },
    # Sample 8
    {
        "Algorithm Used": "Genetic Algorithm",
        "Memory Depth": 1,
        "Mutation Rate": 0.05,
        "Opponent_Name": "Extort2",
        "Coop Rate": 0.95
    },
    # Sample 9
    {
        "Algorithm Used": "Simulated Annealing",
        "Memory Depth": 4,
        "Mutation Rate": 0.1,
        "Opponent_Name": "Prober",
        "Coop Rate": 0.85
    },
    # Sample 10
    {
        "Algorithm Used": "Hill Climbing",
        "Memory Depth": 4,
        "Mutation Rate": 0.01,
        "Opponent_Name": "WSLS",
        "Coop Rate": 0.7
    },
    {
        "Algorithm Used": "Tabular Seach",
        "Memory Depth": 5,
        "Mutation Rate": 0.1,
        "Opponent_Name": "GrimTrigger",
        "Coop Rate": 0.62
    },
    {
        "Algorithm Used": "Tabular Seach",
        "Memory Depth": 2,
        "Mutation Rate": 0.05,
        "Opponent_Name": "ZDGTFT2",
        "Coop Rate": 0.8
    }
]


# Convert sample_inputs to a DataFrame.
sample_df = pd.DataFrame(sample_inputs)

# Reindex sample_df to have the same columns as X
sample_df = sample_df.reindex(columns=X.columns, fill_value="missing")

# Apply safe_transform for non-numeric columns
for col in non_numeric_cols:
    if col in sample_df.columns and col in encoders:
        sample_df[col] = sample_df[col].apply(lambda x: safe_transform(x, encoders[col]))

# Scale the sample inputs using the previously fitted scaler
sample_scaled = scaler.transform(sample_df)

# Predict using each model
pred_lr = log_reg.predict(sample_scaled)
pred_tree = tree.predict(sample_scaled)
pred_rf = rf.predict(sample_scaled)
pred_svm = svm.predict(sample_scaled)

# Print predictions for each sample
print("Predictions for sample inputs:")
for i in range(len(sample_df)):
    print(f"Sample {i+1}: {sample_inputs[i]}")
    print("  Logistic Regression Prediction:", pred_lr[i])
    print("  Decision Tree Prediction:", pred_tree[i])
    print("  Random Forest Prediction:", pred_rf[i])
    print("  SVM Prediction:", pred_svm[i])
    print("")
