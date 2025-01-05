# Logistic Regression

## What
- **Logistic Regression**: A statistical method for binary classification.
- Predicts the probability of class membership using a sigmoid function.
- Outputs probabilities between 0 and 1.

---

## Why
- **Efficient**: Works well with linearly separable data.
- **Interpretable**: Coefficients indicate the impact of features.
- Handles **binary classification** tasks effectively.

---

## How
1. Compute the weighted sum of inputs:
   $$ z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b $$
2. Apply the sigmoid activation function:
   $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
3. Predict the class based on a threshold (e.g., 0.5):
   $$ \hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \geq 0.5 \\ 0 & \text{otherwise} \end{cases} $$
4. Optimize weights using **cross-entropy loss**:
   $$ L = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

---

## When (to use)
- For **binary classification** tasks.
- When the relationship between features and target is **linear**.
- Works well with **small to medium datasets**.

---

## Assumptions
- **Linear relationship** between input features and log-odds of the target.
- Features are **independent** and have minimal multicollinearity.
- The dataset is **balanced** or class imbalance is handled.

---

## Example
- **Task**: Predict if a customer will purchase a product (Yes/No).
- **Dataset**:
  - Age: [22, 25, 47, 52, 46]
  - Salary: [50000, 52000, 110000, 150000, 105000]
  - Purchased: [0, 0, 1, 1, 1]
- **Steps**:
  - Fit a logistic regression model.
  - Use the model to calculate probabilities of purchase.
  - Predict `1` if probability ≥ 0.5, otherwise `0`.

---

## Mathematics
1. **Sigmoid Function**:
   $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

2. **Log-Odds**:
   $$ \log\left(\frac{P(y=1)}{P(y=0)}\right) = w_1x_1 + w_2x_2 + \dots + b $$

3. **Cross-Entropy Loss**:
   $$ L = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

4. **Gradient Descent** (update weights):
   $$ w_j = w_j - \eta \frac{\partial L}{\partial w_j} $$

---

## Code
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Sample Data
X = np.array([[22, 50000], [25, 52000], [47, 110000], [52, 150000], [46, 105000]])
y = np.array([0, 0, 1, 1, 1])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)
```

---

## Key Points to Remember
- **Sigmoid function** maps outputs to probabilities.
- Works best for **linearly separable data**.
- Use **regularization** (e.g., L1, L2) to avoid overfitting.
- Handles binary classification but can be extended to multi-class (e.g., One-vs-Rest).

---

## Extensions
- **Multinomial Logistic Regression**: For multi-class classification.
- **Regularized Logistic Regression**: Adds penalties (L1/L2) to the loss function.
- **Stochastic Logistic Regression**: Uses stochastic gradient descent for optimization.

---

## Pros
- **Simple** and easy to implement.
- **Interpretable coefficients**.
- Performs well with linearly separable data.

---

## Cons
- Assumes **linear decision boundary**.
- Sensitive to **outliers**.
- Requires **balanced datasets** or adjustment for imbalance.

---

## Applications
- **Binary classification** (e.g., spam detection, churn prediction).
- **Medical diagnosis** (e.g., disease presence).
- **Marketing** (e.g., lead conversion prediction).

---

## Limitations
- Struggles with **non-linear relationships**.
- Not suitable for datasets with **high dimensionality** or severe multicollinearity.
- Sensitive to improperly scaled features; requires normalization or standardization.

---