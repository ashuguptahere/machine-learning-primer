# Gradient Boosting

## What
- **Gradient Boosting** is an ensemble learning technique that builds a strong model by sequentially adding weak learners (usually decision trees).
- It optimizes the model by minimizing a loss function using gradient descent.

## Why
- Improves prediction accuracy for both regression and classification.
- Reduces bias and variance, leading to better generalization.
- Works well with structured/tabular data.

## How
1. Initialize the model with a constant prediction (e.g., mean for regression, log-odds for classification).
2. Iteratively:
   - Compute the residuals (errors) based on the current model.
   - Fit a weak learner (e.g., decision tree) to predict the residuals.
   - Update the model by adding the predictions of the weak learner, scaled by a learning rate.
3. Stop after a predefined number of iterations or when performance plateaus.

---

## Assumptions
- Weak learners are slightly better than random guessing.
- Data relationships can be captured by additive models.
- Loss function is differentiable.

---

## Example

### Intuition
- Predict house prices based on **size** and **location**.
- Start with an initial guess (e.g., average price).
- Add trees iteratively to correct prediction errors.

### Data
```plaintext
| Size  | Location | Price  |
|-------|----------|--------|
| 1000  | Urban    | 300000 |
| 1500  | Suburban | 400000 |
| 2000  | Rural    | 250000 |
```

### Process
1. Initialize with the mean price: $350000$.
2. Compute residuals (e.g., actual - predicted).
3. Fit a decision tree to residuals.
4. Update predictions: $NewPrediction = OldPrediction + LearningRate \times TreePrediction$.
5. Repeat until convergence.

---

## Mathematics
### Residuals
- Residual: $r_i = y_i - \hat{y}_i$

### Loss Function
- Common loss functions:
  - Regression: Mean Squared Error (MSE): $L = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$
  - Classification: Log Loss: $L = -\frac{1}{n} \sum [y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i)]$

### Update Rule
- $\hat{y}_{new} = \hat{y}_{old} + \eta \cdot h(x)$
  - $\eta$: Learning rate.
  - $h(x)$: Prediction from the weak learner.

---

## Code
### Regression
```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Sample data
X = np.array([[1000], [1500], [2000]])  # Size of house
y = np.array([300000, 400000, 250000])  # Price

# Model
reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
reg.fit(X, y)

# Prediction
query = np.array([[1200]])
print(reg.predict(query))
```

### Classification
```python
from sklearn.ensemble import GradientBoostingClassifier

# Sample data
X = np.array([[1, 1], [2, 1], [1, 2], [2, 2]])  # Features
y = np.array([0, 0, 1, 1])  # Labels

# Model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
clf.fit(X, y)

# Prediction
query = np.array([[1.5, 1.5]])
print(clf.predict(query))
```

---

## Key Points to Remember
- **Learning Rate**: Controls the contribution of each weak learner. Smaller values improve generalization but require more iterations.
- **Number of Trees**: Too many trees can overfit; use early stopping or cross-validation.
- **Tree Depth**: Shallow trees (weak learners) are preferred.

---

## Extensions
- **XGBoost**: Optimized implementation with regularization.
- **LightGBM**: Faster training using histogram-based learning.
- **CatBoost**: Handles categorical features efficiently.

---

## Pros
- High accuracy for structured data.
- Handles missing data well (with specific implementations).
- Flexible with customizable loss functions.

---

## Cons
- Computationally expensive for large datasets.
- Sensitive to hyperparameters (learning rate, tree depth, etc.).
- Requires careful tuning to avoid overfitting.

---

## Applications
- Predicting house prices.
- Fraud detection.
- Customer churn prediction.

---

## Limitations
- Poor performance with sparse or high-dimensional data.
- Does not perform well on unbalanced datasets without modifications.
- Training can be slow, especially with large datasets.

---