# Bagging (Bootstrap Aggregating)

## What
- **Bagging**: An ensemble method that combines predictions of multiple models to improve stability and accuracy.
- Works by creating multiple subsets of the training data (with replacement) and training a separate model on each subset.
- The final prediction is aggregated (e.g., by averaging for regression or majority voting for classification).

## Why
- Reduces variance and overfitting in high-variance models.
- Increases prediction stability.
- Enhances the performance of weak learners.

## How
1. Generate multiple bootstrap samples from the original dataset.
2. Train a separate model on each bootstrap sample.
3. Combine predictions from all models using aggregation methods:
   - **Regression**: Average predictions.
   - **Classification**: Majority voting.

## When (to use)
- Base model has high variance (e.g., decision trees).
- Dataset is prone to overfitting.
- Need robust predictions.

## Assumptions
- Models are independent and have low bias.
- Dataset has sufficient size for creating diverse bootstrap samples.

## Example

**Dataset**
| Sample | Feature 1 | Feature 2 | Target |
|--------|-----------|-----------|--------|
| A      | 2.5       | 3.1       | 1      |
| B      | 1.2       | 0.7       | 0      |
| C      | 4.5       | 2.2       | 1      |
| D      | 3.1       | 3.9       | 1      |
| E      | 0.5       | 0.6       | 0      |

**Intuition**
- Bootstrap Sample 1: {A, B, D}
- Bootstrap Sample 2: {B, C, E}
- Train separate models on each sample and aggregate predictions.

## Mathematics
1. **Bootstrap Sampling**:
   $$X_b \sim \text{Random Sampling with Replacement}(X)$$

2. **Aggregate Predictions**:
   - Regression:
     $$\hat{y} = \frac{1}{n} \sum_{i=1}^n \hat{y}_i$$
   - Classification:
     $$\hat{y} = \text{Mode}(\hat{y}_1, \hat{y}_2, \dots, \hat{y}_n)$$

## Code
```python
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging Classifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)

# Predictions
y_pred = bagging.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

## Key Points to Remember
- Bagging reduces variance by averaging predictions.
- Works best with high-variance models like decision trees.
- Bootstrap sampling ensures diversity in training subsets.

## Extensions
- **Random Forest**: Bagging applied to decision trees with feature subset selection.
- **Pasting**: Similar to bagging but uses sampling without replacement.

## Pros
- Reduces overfitting.
- Improves prediction stability.
- Handles high-dimensional data well.

## Cons
- Requires more computational resources (multiple models).
- Does not reduce bias (relies on low-bias base models).

## Applications
- Classification and regression tasks.
- Improving model robustness on noisy datasets.
- Reducing overfitting in decision trees.

## Limitations
- Ineffective if the base model has high bias.
- Computational cost increases with the number of models.

---