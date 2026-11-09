
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load and clean the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

df = df.dropna(subset=["Quantity", "UnitPrice"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Use a small sample because hierarchical clustering
# can require a lot of memory
df = df.head(1000).copy()

# Select and scale features
X = df[["Quantity", "UnitPrice"]]
X_scaled = StandardScaler().fit_transform(X)

# Apply hierarchical clustering
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
df["Cluster"] = model.fit_predict(X_scaled)

print("Hierarchical Clustering Results")
print("--------------------------------")
print(df[["Quantity", "UnitPrice", "Cluster"]].head(10))
print("\nCluster Counts:")
print(df["Cluster"].value_counts().sort_index())

if df["Cluster"].nunique() > 1:
    print("\nSilhouette Score:",
          round(silhouette_score(X_scaled, df["Cluster"]), 4))

df.to_csv("hierarchical_results.csv", index=False)
print("\nResults saved to hierarchical_results.csv")
