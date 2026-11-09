
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# STEP 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"

if not os.path.exists(file_path):
    print("ERROR: Dataset file not found!")
    print("Please check this path:", file_path)
    raise SystemExit

data = pd.read_csv(file_path, encoding="ISO-8859-1")

print("Dataset loaded successfully!")
print("Dataset size:", data.shape)
print("\nFirst five rows:")
print(data.head())

# STEP 2: Check required columns
required_columns = [
    "InvoiceNo", "Description", "Quantity", "Country"
]

missing_columns = [
    col for col in required_columns if col not in data.columns
]

if missing_columns:
    print("\nMissing columns:", missing_columns)
    print("Available columns:", data.columns.tolist())
    raise SystemExit

# STEP 3: Clean the dataset
data = data.dropna(subset=["InvoiceNo", "Description"])
data = data[data["Quantity"] > 0]

# Remove cancelled invoices
data = data[
    ~data["InvoiceNo"].astype(str).str.startswith("C")
]

# Select transactions from the United Kingdom
data = data[data["Country"] == "United Kingdom"]

# Limit the data to 5000 invoices to make processing manageable
invoice_list = data["InvoiceNo"].drop_duplicates().head(5000)
data = data[data["InvoiceNo"].isin(invoice_list)]

# STEP 4: Create a list of products for each invoice
transactions = (
    data.groupby("InvoiceNo")["Description"]
    .apply(lambda products: list(set(products)))
)

# Keep transactions containing at least two different products
transactions = transactions[
    transactions.apply(len) >= 2
].tolist()

print("\nNumber of transactions:", len(transactions))

if not transactions:
    print("ERROR: No suitable transactions found.")
    raise SystemExit

# STEP 5: Convert transactions into a one-hot encoded table
encoder = TransactionEncoder()

encoded_array = encoder.fit(transactions).transform(
    transactions, sparse=True
)

basket = pd.DataFrame.sparse.from_spmatrix(
    encoded_array,
    columns=encoder.columns_
)

print("Transaction table size:", basket.shape)

# STEP 6: Apply Apriori algorithm
frequent_items = apriori(
    basket,
    min_support=0.02,
    use_colnames=True,
    max_len=3,
    low_memory=True
)

print("\n========== FREQUENT ITEMSETS ==========")

if frequent_items.empty:
    print("No frequent itemsets found.")
    print("Try reducing min_support to 0.01.")
    raise SystemExit

print(frequent_items.head(10).to_string(index=False))

# STEP 7: Generate association rules
print("\n========== ASSOCIATION RULES ==========")

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.3
)

if rules.empty:
    print("No association rules found.")
else:
    rules = rules.sort_values(
        by="confidence", ascending=False
    )

    columns_to_show = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]

    print(
        rules[columns_to_show]
        .head(10)
        .to_string(index=False)
    )

# STEP 8: Save the results
frequent_items.to_csv(
    "apriori_frequent_itemsets.csv", index=False
)

rules.to_csv(
    "apriori_association_rules.csv", index=False
)

print("\n========== COMPLETED ==========")
print("Frequent itemsets saved successfully.")
print("Association rules saved successfully.")
