# Linear Regression Notes

## What
- **Definition**: A statistical method to model the relationship between a dependent variable (target) and one or more independent variables (features).
- **Type**: Supervised learning algorithm (Regression).
- **Model**: Predicts the value of the dependent variable as a linear combination of independent variables.

---

## Why
- **Interpretability**: Simple and easy to understand.
- **Speed**: Computationally efficient for small to medium datasets.
- **Use Cases**: Useful when the relationship between variables is linear.

---

## How
1. **Formulate the equation**:
   - $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n + \epsilon$
   - $y$: Dependent variable
   - $x_i$: Independent variables
   - $\beta_i$: Coefficients (parameters to learn)
   - $\epsilon$: Error term (assumed to be normally distributed)

2. **Optimize the coefficients**:
   - Minimize the Residual Sum of Squares (RSS):
     $$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

3. **Make Predictions**:
   - Use the learned coefficients to predict new data: $\hat{y} = \beta_0 + \sum_{i=1}^n \beta_i x_i$

---

## Assumptions
- **Linearity**: The relationship between dependent and independent variables is linear.
- **Independence**: Observations are independent.
- **Homoscedasticity**: Constant variance of residuals.
- **Normality**: Residuals are normally distributed.
- **No multicollinearity**: Independent variables are not highly correlated.

---

## Example

Predict house prices based on square footage.

| Square Footage | Price (in $1000) |
|----------------|------------------|
| 1000           | 150              |
| 1500           | 200              |
| 2000           | 250              |
| 2500           | 300              |

### Steps:
1. **Collect Data**:
   - Features: Square footage ($x$).
   - Target: House price ($y$).

2. **Visualize Relationship**:
   - Scatter plot: House price vs. square footage.

3. **Fit a Line**:
   - Find the best-fit line: $y = \beta_0 + \beta_1 x$.

4. **Prediction**:
   - Given $x = 2000$ (square footage), predict $y$ (house price).

---

## Mathematics of Linear Regression
1. **Hypothesis Function**:
   $$h(\mathbf{x}) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n$$

2. **Cost Function**:
   $$J(\beta_0, \beta_1, \ldots, \beta_n) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$

3. **Gradient Descent**:
   - Update rule:
     $$\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}$$
   - $\alpha$: Learning rate

---

## Code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Data
data = {
    'SquareFootage': [1500, 2000, 2500, 3000, 3500],
    'Price': [300000, 400000, 500000, 600000, 700000]
}
df = pd.DataFrame(data)

# Features and Target
X = df[['SquareFootage']]
y = df['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.legend()
plt.show()
```

---

## Key Points to Remember
- Linear regression is sensitive to outliers.
- Assumes linear relationships; won't work well for non-linear patterns.
- Feature scaling isn't required for linear regression.

---

## Extensions
- **Polynomial Regression**: Add polynomial features for non-linear relationships.
- **Ridge Regression**: Adds L2 regularization to reduce overfitting.
- **Lasso Regression**: Adds L1 regularization for feature selection.
- **ElasticNet**: Combines L1 and L2 regularization.

---

## Pros:
- Easy to implement.
- Interpretability.
- Efficient for small datasets.

---

## Cons:
- Poor performance with non-linear data.
- Assumes independence and linearity.
- Sensitive to outliers.

---

## Applications
- Predicting house prices.
- Estimating sales revenue.
- Risk assessment in finance.

---

## Limitations
- Assumes linear relationship.
- Can't handle missing data or categorical variables directly.
- Performance depends on meeting assumptions.

---