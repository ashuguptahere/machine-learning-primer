# Decision Tree

## What
- A **Decision Tree** is a tree-like structure used for decision-making and prediction.
- It splits data into branches based on feature values to predict the target variable.
- Types of decision trees:
  - **Classification Tree**: Predicts categorical outcomes.
  - **Regression Tree**: Predicts continuous outcomes.

## Why
- Simple to understand and interpret.
- Handles both numerical and categorical data.
- Useful for non-linear relationships between features and targets.

## How
- Start at the root node (entire dataset).
- Split data based on a feature to minimize impurity:
  - **Impurity Metrics**:
    - **Gini Index** (Classification): $G = 1 - \sum_{i} p_i^2$
    - **Entropy** (Classification): $H = -\sum_{i} p_i \log_2(p_i)$
    - **Variance Reduction** (Regression): $Var = \text{Variance before split} - \text{Variance after split}$
- Repeat recursively for child nodes until:
  - Leaf node is pure.
  - Maximum depth is reached.
  - Minimum samples per node condition is met.

---

## Assumptions
- Features are independent.
- Homogeneous groups in target variable can be formed using splits.

---

## Example

### Intuition
- Data: Predict if a person buys a product (Yes/No) based on **Age** and **Income**.
- Initial split:
  - Age ≤ 30 → Further split.
  - Age > 30 → Classify as "No" (pure node).

### Data
```plaintext
| Age  | Income  | Buys |
|------|---------|------|
| 25   | High    | No   |
| 30   | Medium  | Yes  |
| 35   | Low     | Yes  |
```

### Process
1. Calculate impurity for splits.
2. Choose the best feature to split.
3. Split data into branches.
4. Repeat for child nodes.

---

## Mathematics
### Gini Index
- Formula: $G = 1 - \sum_{i=1}^{n} p_i^2$
- Example:
  - Node: [3 Yes, 2 No]
  - $p_{Yes} = \frac{3}{5}, p_{No} = \frac{2}{5}$
  - $G = 1 - (\frac{3}{5})^2 - (\frac{2}{5})^2 = 0.48$

### Entropy
- Formula: $H = -\sum_{i=1}^{n} p_i \log_2(p_i)$
- Example:
  - Node: [3 Yes, 2 No]
  - $p_{Yes} = \frac{3}{5}, p_{No} = \frac{2}{5}$
  - $H = -(\frac{3}{5}\log_2\frac{3}{5} + \frac{2}{5}\log_2\frac{2}{5}) = 0.97$

### Information Gain
- $IG = H_{parent} - \sum \frac{|Node|}{|Parent|} H_{Node}$

---

## Code
### Classification Tree
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Sample data
X = [[25, 1], [30, 2], [35, 0]]  # Age, Income encoded as numerical
y = [0, 1, 1]  # Buys: No (0), Yes (1)

# Model
clf = DecisionTreeClassifier(criterion="gini", max_depth=3)
clf.fit(X, y)

# Visualization
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=["Age", "Income"], class_names=["No", "Yes"], filled=True)
plt.show()
```

### Regression Tree
```python
from sklearn.tree import DecisionTreeRegressor

# Sample data
X = [[1], [2], [3], [4], [5]]  # Feature
y = [5, 7, 9, 11, 13]          # Target

# Model
reg = DecisionTreeRegressor(max_depth=2)
reg.fit(X, y)

# Prediction
print(reg.predict([[6]]))
```

---

## Key Points to Remember
- **Splitting Criteria**:
  - Gini Index: Faster, commonly used.
  - Entropy: Information-theoretic.
- **Overfitting**: Use max depth, min samples, or pruning.
- **Interpretability**: Easy to visualize.

---

## Extensions
- **Random Forest**: Ensemble of multiple decision trees.
- **Gradient Boosted Trees**: Sequentially build trees to minimize error.

---

## Pros
- Simple to interpret and explain.
- Handles mixed data types.
- No need for feature scaling.

---

## Cons
- Prone to overfitting without constraints.
- Sensitive to small data changes.
- Does not generalize well on continuous data.

---

## Applications
- Medical diagnosis.
- Loan approval prediction.
- Customer segmentation.

---

## Limitations
- Instability: Small changes in data can alter the tree.
- Biased towards dominant classes.
- Can be computationally expensive for large datasets.

---