# Polynomial Regression

## What
- Polynomial Regression is a type of regression analysis where the relationship between the independent variable $x$ and the dependent variable $y$ is modeled as an $n^{th}$ degree polynomial.

---

## Why
- Captures non-linear relationships that cannot be modeled by simple linear regression.
- Useful when the data exhibits curvatures.

---

## How
- Extend linear regression by including polynomial terms (e.g., $x^2$, $x^3$) as additional features.
- Fit the polynomial model to minimize the error.

---

## Assumptions
- The relationship between the variables can be well-approximated by a polynomial.
- Errors are normally distributed.
- Homoscedasticity: Variance of errors is constant across all values of $x$.

---

## Example

### Intuition
- Imagine modeling house prices based on area.
- The relationship isn't linear: small houses and large houses may have different pricing dynamics.
- A polynomial regression can capture these patterns better than a straight line.

### Dataset
| Area (sq ft) | Price ($\times 10^3$) |
|--------------|-----------------------|
| 600          | 200                   |
| 800          | 300                   |
| 1000         | 450                   |
| 1200         | 500                   |
| 1400         | 700                   |

---

## Mathematics
- Polynomial model:  
  $$y = \beta_0 + \beta_1x + \beta_2x^2 + \cdots + \beta_nx^n + \epsilon$$
- Cost function (mean squared error):  
  $$J(\beta) = \frac{1}{2m} \sum_{i=1}^m \left( y_i - \hat{y}_i \right)^2$$
- Gradient Descent Update Rule:  
  $$\beta_j \leftarrow \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}$$

---

## Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Data
X = np.array([600, 800, 1000, 1200, 1400]).reshape(-1, 1)
y = np.array([200, 300, 450, 500, 700])

# Polynomial Features (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_poly, y)

# Predictions
y_pred = model.predict(X_poly)

# Plot
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Polynomial Fit')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price (x 10^3)')
plt.legend()
plt.show()

# Mean Squared Error
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
```

---

## Key Points to Remember
- Polynomial Regression extends Linear Regression by adding polynomial features.
- Overfitting is a risk with high-degree polynomials.
- Always preprocess features for better results.

---

## Extensions
- Ridge Regression for Polynomial Features to reduce overfitting.
- Use Cross-Validation to select the degree of the polynomial.
- Polynomial Kernels in Support Vector Machines (SVMs).

---

## Pros
- Captures non-linear patterns effectively.
- Simple extension of linear regression.

---

## Cons
- Prone to overfitting with high-degree polynomials.
- Sensitive to outliers.
- Model interpretation becomes difficult as degree increases.

---

## Applications
- Modeling growth curves in biology.
- Predicting pricing trends.
- Forecasting and trend analysis in finance.
- Analyzing experimental data with curvature.

---

## Limitations
- Limited to polynomial relationships.
- Requires careful tuning of the polynomial degree.
- High-degree models can become computationally expensive.
- May perform poorly on extrapolated data.

---