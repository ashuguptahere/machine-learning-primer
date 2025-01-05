# AdaBoost (Adaptive Boosting)

## What
- **AdaBoost** is an ensemble learning algorithm that combines multiple weak learners (usually decision stumps) to form a strong learner.
- Boosts the performance of weak classifiers by focusing on incorrectly classified samples.

## Why
- Improves prediction accuracy over individual weak learners.
- Reduces bias and variance compared to standalone models.
- Works well with a variety of weak learners.

## How
1. Initialize sample weights uniformly.
2. For each weak learner:
   - Train a weak learner on weighted samples.
   - Compute the error rate $e$: proportion of misclassified samples.
   - Compute the model weight $\alpha$: $\alpha = \frac{1}{2} \ln\left(\frac{1-e}{e}\right)$.
   - Update sample weights: Increase weights for misclassified samples.
3. Combine all weak learners weighted by their $\alpha$ values.

---

## Assumptions
- Weak learners perform slightly better than random guessing.
- Data is clean and free of significant noise or outliers.

---

## Example

### Intuition
- Combine several weak classifiers (e.g., decision stumps) to classify points.
- Focus more on misclassified points in each iteration to improve performance.

### Data
```plaintext
| Feature 1 | Feature 2 | Label |
|-----------|-----------|-------|
| 1         | 1         | 1     |
| 2         | 1         | 1     |
| 1         | 2         | -1    |
| 2         | 2         | -1    |
```

### Process
1. Initialize weights for all samples as $\frac{1}{n}$.
2. Train a decision stump and compute error.
3. Update weights based on misclassifications.
4. Repeat for multiple iterations.

---

## Mathematics
### Model Weight ($\alpha$)
- $\alpha = \frac{1}{2} \ln\left(\frac{1-e}{e}\right)$, where $e$ is the weighted error rate of the weak learner.

### Weight Update
- $w_i \leftarrow w_i \times \exp(\alpha \times I[y_i \neq h(x_i)])$
- Normalize weights: $w_i = \frac{w_i}{\sum w_i}$

### Final Prediction
- $H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$

---

## Code
### Classification
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Sample data
X = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])
y = np.array([1, 1, -1, -1])

# Model
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)
clf.fit(X, y)

# Prediction
query = np.array([[1.5, 1.5]])
print(clf.predict(query))
```

### Regression
```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([1.5, 2.0, 3.5, 4.0])

# Model
reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1), n_estimators=50)
reg.fit(X, y)

# Prediction
query = np.array([[2.5]])
print(reg.predict(query))
```

---

## Key Points to Remember
- **Weak Learner**: Usually decision stumps in AdaBoost.
- **Weights**: Adjust sample weights to focus on misclassified samples.
- **Final Model**: Weighted combination of all weak learners.

---

## Extensions
- **Gradient Boosting**: Generalization using gradient descent to optimize loss.
- **XGBoost**: Optimized implementation of gradient boosting.
- **LightGBM**: Gradient boosting with histogram-based learning.

---

## Pros
- Boosts performance of weak learners.
- Robust to overfitting with proper tuning.
- Handles both classification and regression tasks.

---

## Cons
- Sensitive to noisy data and outliers.
- Slower training due to iterative process.
- Does not scale well with very large datasets.

---

## Applications
- Face detection.
- Fraud detection.
- Sentiment analysis.

---

## Limitations
- Prone to overfitting if weak learners are too complex.
- Requires careful tuning of the number of estimators and learning rate.
- Performance drops with noisy or imbalanced data.

---