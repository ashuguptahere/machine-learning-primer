# Random Forest

## What
- **Definition**: An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
- **Purpose**:
  - Handle both classification and regression tasks.
  - Improve the robustness and generalization of models.
- **Key Idea**: Uses bagging (Bootstrap Aggregation) to create multiple decision trees on different subsets of data and aggregates their predictions.

---

## Key Concepts

### 1. Bagging
- **Definition**: A technique to reduce variance by training models on different bootstrap samples of data.
- Steps:
  1. Create random subsets of the dataset with replacement.
  2. Train a decision tree on each subset.
  3. Aggregate predictions (majority vote for classification, average for regression).

### 2. Feature Randomness
- **Definition**: At each split in a tree, a random subset of features is considered.
- **Purpose**:
  - Reduces correlation between trees.
  - Increases diversity of models.

### 3. Out-of-Bag (OOB) Error
- **Definition**: Error estimated using data not included in the bootstrap sample.
- **Purpose**:
  - Acts as a built-in validation set.
  - Reduces the need for cross-validation.

---

## Mathematical

### For Classification
- **Final Prediction**: Aggregates predictions from all trees using majority voting:
  $$
  \hat{y} = 	ext{mode} \{ T_1(X), T_2(X), \dots, T_n(X) \}
  $$
  - $ T_i(X) $: Prediction from the $ i $-th tree.

### For Regression
- **Final Prediction**: Takes the average of predictions from all trees:
  $$
  \hat{y} = \frac{1}{n} \sum_{i=1}^n T_i(X)
  $$
  - $ T_i(X) $: Prediction from the $ i $-th tree.

---

## Code

### Random Forest for Classification (Scikit-Learn)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
clf.fit(X_train, y_train)

# Predictions
predictions = clf.predict(X_test)
print("Predictions:", predictions)

# OOB Score
print("OOB Score:", clf.oob_score_)
```

### Random Forest for Regression (Scikit-Learn)
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Data
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([5, 7, 9, 11, 13])          # Target

# Model
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X, y)

# Predictions
predictions = reg.predict([[2.5], [4.5]])
print("Predictions:", predictions)
```

---

## Advantages
- Handles both classification and regression tasks.
- Reduces overfitting by averaging multiple trees.
- Provides feature importance scores.
- Robust to noisy data and outliers.

## Disadvantages
- Can be computationally expensive (many trees).
- Less interpretable than individual decision trees.
- Requires careful tuning of hyperparameters (e.g., number of trees, max depth).

---

## Key Hyperparameters
- **n_estimators**: Number of trees in the forest.
- **max_depth**: Maximum depth of each tree.
- **max_features**: Maximum number of features considered at each split.
- **min_samples_split**: Minimum samples required to split an internal node.
- **oob_score**: Whether to use out-of-bag samples for validation.

---