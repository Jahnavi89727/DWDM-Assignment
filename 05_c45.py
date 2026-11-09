
import pandas as pd
import numpy as np
from math import log2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Step 2: Clean the data
df = df.dropna(subset=["Quantity", "UnitPrice", "Country"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df.head(10000).copy()

# Step 3: Create target classes
median_quantity = df["Quantity"].median()
df["QuantityClass"] = np.where(
    df["Quantity"] > median_quantity, "High", "Low"
)

# Step 4: Create categorical features
df["PriceCategory"] = pd.qcut(
    df["UnitPrice"].rank(method="first"),
    q=3,
    labels=["Low", "Medium", "High"]
)

X = df[["Country", "PriceCategory"]].astype(str)
y = df["QuantityClass"]

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 6: Calculate entropy
def entropy(values):
    probabilities = values.value_counts(normalize=True)
    return -sum(p * log2(p) for p in probabilities if p > 0)

# Step 7: Calculate gain ratio
def gain_ratio(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = 0
    split_info = 0

    for value, group in data.groupby(feature, observed=True):
        probability = len(group) / len(data)
        weighted_entropy += probability * entropy(group[target])

        if probability > 0:
            split_info -= probability * log2(probability)

    information_gain = total_entropy - weighted_entropy

    if split_info == 0:
        return 0

    return information_gain / split_info

# Step 8: Display gain ratio of each feature
train_data = X_train.copy()
train_data["QuantityClass"] = y_train

print("C4.5-Style Decision Tree Results")
print("--------------------------------")
print("\nGain Ratio:")
for feature in X_train.columns:
    print(feature, ":", round(
        gain_ratio(train_data, feature, "QuantityClass"), 4
    ))

# Step 9: Train a practical tree model
# Entropy is used here; this is not a full C4.5 implementation.
X_encoded = pd.get_dummies(X, dtype=int)

X_train_encoded = X_encoded.loc[X_train.index]
X_test_encoded = X_encoded.loc[X_test.index]

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)
model.fit(X_train_encoded, y_train)

# Step 10: Predict and evaluate
y_pred = model.predict(X_test_encoded)

print("\nAccuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 11: Save results
results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred
results.to_csv("c45_predictions.csv", index=False)

print("\nResults saved to c45_predictions.csv")
