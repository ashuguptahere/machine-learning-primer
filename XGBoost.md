# XGBoost (Extreme Gradient Boosting)

## What
- **XGBoost** is an optimized implementation of Gradient Boosting designed for speed and performance.
- Combines decision trees to create a strong predictive model using parallel processing, regularization, and efficient memory usage.

## Why
- Faster and more efficient than standard Gradient Boosting.
- Includes regularization to reduce overfitting.
- Supports missing values and sparse data.

## How
1. Initialize with a base prediction.
2. Iteratively fit weak learners (decision trees) to the negative gradient of the loss function.
3. Use advanced techniques like:
   - Regularization (L1/L2).
   - Weighted quantile sketch for approximate tree building.
   - Column subsampling to reduce overfitting.
4. Optimize the objective function using second-order Taylor approximation.

---

## Assumptions
- Weak learners can improve over random guessing.
- Data relationships can be captured by additive models.
- Loss function is differentiable.

---

## Example

### Intuition
- Predict whether a customer will churn based on their activity.
- Use multiple iterations of trees to minimize prediction errors.

### Data
```plaintext
| Feature 1 | Feature 2 | Churn |
|-----------|-----------|-------|
| 10        | 100       | 1     |
| 15        | 80        | 0     |
| 8         | 120       | 1     |
| 20        | 90        | 0     |
```

### Process
1. Start with an initial guess for the churn probability.
2. Fit trees sequentially to minimize the log-loss error.
3. Combine the predictions of all trees.

---

## Mathematics
### Objective Function
- $Obj = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{t=1}^T \Omega(f_t)$
  - $L$ is the loss function (e.g., log-loss for classification).
  - $\Omega(f_t)$ is the regularization term: $\Omega(f_t) = \frac{1}{2} \lambda \|w\|^2 + \gamma T$.

### Update Rule
- Use second-order Taylor expansion:
  - $Obj \approx \sum g_i \Delta \hat{y}_i + \frac{1}{2} h_i \Delta \hat{y}_i^2 + \Omega(f_t)$
  - $g_i$: First-order gradient.
  - $h_i$: Second-order gradient (Hessian).

---

## Code
### Classification
```python
from xgboost import XGBClassifier
import numpy as np

# Sample data
X = np.array([[10, 100], [15, 80], [8, 120], [20, 90]])
y = np.array([1, 0, 1, 0])

# Model
clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1)
clf.fit(X, y)

# Prediction
query = np.array([[12, 95]])
print(clf.predict(query))
```

### Regression
```python
from xgboost import XGBRegressor

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([1.5, 2.0, 3.5, 4.0])

# Model
reg = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1)
reg.fit(X, y)

# Prediction
query = np.array([[2.5]])
print(reg.predict(query))
```

---

## Key Points to Remember
- **Regularization**: L1 (lasso) and L2 (ridge) are used to prevent overfitting.
- **Feature Importance**: Provides insights into the most relevant features.
- **Handling Missing Data**: Automatically handles missing values.

---

## Extensions
- **LightGBM**: Faster, optimized for large datasets.
- **CatBoost**: Efficient with categorical features.
- **XGBoost GPU**: Leverages GPU for faster training.

---

## Pros
- High performance and accuracy.
- Built-in regularization reduces overfitting.
- Flexible with customizable loss functions.

---

## Cons
- Requires careful tuning of hyperparameters.
- Computationally expensive for very large datasets.
- Sensitive to noise in data.

---

## Applications
- Fraud detection.
- Customer churn prediction.
- Ranking problems (e.g., search engines).

---

## Limitations
- Slower training compared to simpler models.
- Struggles with very high-dimensional sparse data.
- Overfitting risk with too many trees or complex trees.

---