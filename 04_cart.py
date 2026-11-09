
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Step 2: Clean the data
df = df.dropna(subset=["Quantity", "UnitPrice", "Country"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Step 3: Select a sample
df = df.head(10000).copy()

# Step 4: Create Low and High quantity classes
median_quantity = df["Quantity"].median()

df["QuantityClass"] = df["Quantity"].apply(
    lambda q: "High" if q > median_quantity else "Low"
)

# Step 5: Select features and target
X = df[["UnitPrice", "Country"]]
y = df["QuantityClass"]

# Convert categorical data into numeric columns
X = pd.get_dummies(X, columns=["Country"], dtype=int)

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Build the CART decision tree
model = DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Step 8: Predict the results
y_pred = model.predict(X_test)

# Step 9: Display the results
print("CART Decision Tree Results")
print("--------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nDecision Tree Rules:")
print(export_text(model, feature_names=list(X.columns)))

# Step 10: Save predictions
results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

results.to_csv("cart_predictions.csv", index=False)

print("\nResults saved to cart_predictions.csv")
