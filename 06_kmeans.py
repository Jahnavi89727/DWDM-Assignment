
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Step 2: Clean the data
df = df.dropna(subset=["Quantity", "UnitPrice"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Step 3: Select a sample
df = df.head(5000).copy()

# Step 4: Select numerical features
X = df[["Quantity", "UnitPrice"]]

# Step 5: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Apply K-Means
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

# Step 7: Display results
print("K-Means Clustering Results")
print("--------------------------")
print(df[["Quantity", "UnitPrice", "Cluster"]].head(10))
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

if df["Cluster"].nunique() > 1:
    print("\nSilhouette Score:",
          round(silhouette_score(X_scaled, df["Cluster"]), 4))

# Step 8: Save results
df.to_csv("kmeans_results.csv", index=False)
print("\nResults saved to kmeans_results.csv")
