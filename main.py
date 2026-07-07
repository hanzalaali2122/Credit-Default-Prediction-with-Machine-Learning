import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

try:
    file = pd.read_excel("finance.xlsx")
    print("✅ Dataset Loaded Successfully")
except Exception as e:
    print("❌ Error Loading Dataset")
    print(e)
    exit()

# ============================================================
# DATASET OVERVIEW
# ============================================================

print("\n--- First 5 Rows ---")
print(file.head())

print("\n--- Column Names ---")
print(file.columns)

print("\n--- Dataset Shape ---")
print(file.shape)

# Rename unnamed column if exists
if "Unnamed: 0" in file.columns:
    file.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

# ============================================================
# DATA CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove spaces from column names
file.columns = file.columns.str.strip()

# Clean text columns
file["default"] = file["default"].astype(str).str.strip().str.capitalize()
file["student"] = file["student"].astype(str).str.strip().str.capitalize()

# Missing values
print("\nMissing Values:")
print(file.isnull().sum())

# Remove duplicates
duplicates = file.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

if duplicates > 0:
    file.drop_duplicates(inplace=True)

# Encode categorical columns
file["default_encoded"] = file["default"].map({"Yes": 1, "No": 0})
file["student_encoded"] = file["student"].map({"Yes": 1, "No": 0})

# Check encoding
print("\nEncoded Columns Preview:")
print(file[["default", "default_encoded",
            "student", "student_encoded"]].head())

# ============================================================
# BASIC EDA
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nStatistics:")
print(file[["balance", "income"]].describe())

# Default counts
default_counts = file["default"].value_counts()

print("\nDefault Counts:")
print(default_counts)

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1 Default Count
default_counts.plot(
    kind="bar",
    ax=axes[0, 0],
    color=["steelblue", "tomato"]
)

axes[0, 0].set_title("Default Count")
axes[0, 0].tick_params(axis='x', rotation=0)

# 2 Balance Distribution
axes[0, 1].hist(
    file["balance"],
    bins=30,
    color="skyblue",
    edgecolor="black"
)

axes[0, 1].set_title("Balance Distribution")

# 3 Income Distribution
axes[1, 0].hist(
    file["income"],
    bins=30,
    color="lightgreen",
    edgecolor="black"
)

axes[1, 0].set_title("Income Distribution")

# 4 Scatter Plot
colors = file["default"].map({
    "No": "blue",
    "Yes": "red"
})

axes[1, 1].scatter(
    file["income"],
    file["balance"],
    c=colors,
    alpha=0.5
)

axes[1, 1].set_title("Income vs Balance")

plt.tight_layout()

plt.savefig("eda_visualization.png")

plt.show()

print("✅ Visualization Saved")

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 60)
print("MACHINE LEARNING MODEL")
print("=" * 60)

# Features
X = file[["balance", "income", "student_encoded"]]

# Target
y = file["default_encoded"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\n--- Logistic Regression ---")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Accuracy: {lr_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# ============================================================
# DECISION TREE
# ============================================================

print("\n--- Decision Tree ---")

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"Accuracy: {dt_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY ✅")
print("=" * 60)