from apriori import TransactionDataset, Apriori

# Create a TransactionDataset object with the given transactions
transaction_data = TransactionDataset(transactions=[
    ['butter', 'bread', 'milk', 'sugar'],
    ['butter', 'flour', 'milk', 'sugar'],
    ['butter', 'eggs', 'milk', 'salt'],
    ['eggs'],
    ['butter', 'flour', 'milk', 'salt', 'sugar']
])

# Set the minimum support and minimum confidence values
min_support = 0.5
min_confidence = 0.5

# Create an instance of the Apriori class
apriori_model = Apriori(transaction_dataset=transaction_data, min_support=min_support)

# Run the Apriori algorithm to find frequent itemsets
freq_itemsets = apriori_model.apriori_Algorithm()

# Print the frequent itemsets and their counts
for itemset, count in freq_itemsets.items():
    print(f"Itemset: {itemset}, Count: {count}")

# Generate association rules with the specified minimum confidence
apriori_model.association_rules(min_confidence)

# Print the generated association rules and their confidence
for rule, confidence in apriori_model.rules.items():
    antecedent, consequent = rule
    print(f"Rule: {antecedent} -> {consequent}, Confidence: {confidence}")
