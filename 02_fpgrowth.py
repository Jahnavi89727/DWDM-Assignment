
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\archive\OnlineRetail.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Step 2: Clean the dataset
df = df.dropna(subset=["InvoiceNo", "Description"])
df = df[df["Quantity"] > 0]
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Country"] == "United Kingdom"]

# Step 3: Select the first 5000 invoices
invoices = df["InvoiceNo"].drop_duplicates().head(5000)
df = df[df["InvoiceNo"].isin(invoices)]

# Step 4: Convert invoices into transactions
transactions = (
    df.groupby("InvoiceNo")["Description"]
    .apply(lambda x: list(set(x)))
    .tolist()
)

# Keep transactions containing at least 2 products
transactions = [t for t in transactions if len(t) >= 2]

# Step 5: Convert transactions into a one-hot encoded table
te = TransactionEncoder()
encoded_data = te.fit(transactions).transform(transactions, sparse=True)

basket = pd.DataFrame.sparse.from_spmatrix(
    encoded_data,
    columns=te.columns_
)

print("Total transactions:", len(basket))

# Step 6: Apply FP-Growth
frequent_itemsets = fpgrowth(
    basket,
    min_support=0.02,
    use_colnames=True,
    max_len=3
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))

# Step 7: Generate association rules
if not frequent_itemsets.empty:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.3
    )

    if not rules.empty:
        rules = rules.sort_values(
            by="confidence",
            ascending=False
        )

        print("\nAssociation Rules:")
        print(
            rules[
                ["antecedents", "consequents",
                 "support", "confidence", "lift"]
            ].head(10).to_string(index=False)
        )

        rules.to_csv(
            "fpgrowth_association_rules.csv",
            index=False
        )
    else:
        print("\nNo association rules found.")
else:
    print("\nNo frequent itemsets found.")

# Step 8: Save the frequent itemsets
frequent_itemsets.to_csv(
    "fpgrowth_frequent_itemsets.csv",
    index=False
)

print("\nFP-Growth completed successfully!")
print("Output CSV files saved in the project folder.")
