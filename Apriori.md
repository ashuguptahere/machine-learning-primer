# Apriori Algorithm

## What
- **Apriori**: A rule-based algorithm used for frequent itemset mining and association rule learning.
- Identifies frequent itemsets in transactional datasets and generates association rules.
- Works by applying the **Apriori property**: If an itemset is frequent, all its subsets must also be frequent.

## Why
- Discover relationships or patterns in large transactional datasets.
- Used in market basket analysis to identify items frequently bought together.

## How
1. Define minimum support and confidence thresholds.
2. Identify all itemsets that satisfy the minimum support.
3. Generate candidate itemsets using frequent itemsets (Apriori property).
4. Prune non-frequent itemsets.
5. Generate association rules that satisfy minimum confidence.

## When (to use)
- Analyzing transactional data with discrete categories (e.g., market basket data).
- Identifying co-occurrence relationships among items.
- When you can define meaningful thresholds for support and confidence.

## Assumptions
- Dataset consists of discrete transactions.
- Frequent itemsets exist and can be discovered.
- Thresholds for support and confidence are meaningful for the dataset.

## Example

**Transactional Dataset**
| Transaction ID | Items           |
|----------------|-----------------|
| 1              | Milk, Bread     |
| 2              | Milk, Diaper    |
| 3              | Milk, Bread, Diaper |
| 4              | Bread, Butter   |
| 5              | Milk, Bread, Butter |

**Definitions**
- **Support**:
  $$\text{Support}(A) = \frac{\text{Transactions containing } A}{\text{Total transactions}}$$
- **Confidence**:
  $$\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$

**Steps**
1. Minimum support = 0.6, Minimum confidence = 0.7.
2. Frequent itemsets:
   - {Milk}: 0.8, {Bread}: 0.8, {Milk, Bread}: 0.6
3. Association rules:
   - {Milk} → {Bread}: Confidence = 0.75

## Mathematics
1. **Support**:
   $$\text{Support}(A) = \frac{|\text{Transactions containing } A|}{|\text{Total transactions}|}$$

2. **Confidence**:
   $$\text{Confidence}(A \rightarrow B) = \frac{|\text{Transactions containing } A \cup B|}{|\text{Transactions containing } A|}$$

3. **Lift**:
   $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}$$

## Code
```python
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Sample Data
data = {
    'Milk': [1, 1, 1, 0, 1],
    'Bread': [1, 0, 1, 1, 1],
    'Diaper': [0, 1, 1, 0, 0],
    'Butter': [0, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Frequent Itemsets
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

# Association Rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
print(rules)
```

## Key Points to Remember
- Apriori uses a breadth-first search to find frequent itemsets.
- The **Apriori property** reduces the search space.
- Support, confidence, and lift are key metrics.

## Extensions
- **FP-Growth**: More efficient algorithm for large datasets.
- **Eclat**: Uses depth-first search for frequent itemset mining.

## Pros
- Simple and easy to implement.
- Provides interpretable association rules.
- Works well on small to medium-sized datasets.

## Cons
- Computationally expensive for large datasets.
- Requires discretized data.
- Setting thresholds for support and confidence can be challenging.

## Applications
- Market basket analysis.
- Recommender systems.
- Customer behavior analysis.
- Bioinformatics (gene co-occurrence).

## Limitations
- Inefficient for datasets with many items or low support thresholds.
- Sensitive to noisy or sparse data.
- Does not handle continuous variables without preprocessing.

---