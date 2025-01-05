# Support Vector Machine (SVM)

## What
- **SVM** is a supervised learning algorithm used for classification and regression tasks.
- Finds the optimal hyperplane that maximizes the margin between different classes.
- Can work with both linear and non-linear data.

## Why
- Works well for high-dimensional data.
- Effective in cases where the number of features is greater than the number of samples.
- Flexible with different kernel functions for non-linear decision boundaries.

## How
1. Represent data points in an $n$-dimensional space.
2. Find the hyperplane that best separates the classes by maximizing the margin.
3. Use kernel tricks to handle non-linear data by mapping it to higher dimensions.

---

## Assumptions
- Data is linearly separable (for linear SVM).
- Classes are balanced or close to balanced.
- Misclassification can be controlled using a soft margin parameter ($C$).

---

## Example

### Intuition
- Classify whether a point belongs to Class A or Class B.
- Use a straight line (or curve with kernels) to separate the points.

### Data
```plaintext
| Feature 1 | Feature 2 | Label |
|-----------|-----------|-------|
| 1.0       | 2.0       | 1     |
| 2.0       | 3.0       | 1     |
| 3.0       | 3.0       | 0     |
| 4.0       | 5.0       | 0     |
```

### Process
1. Map data to a high-dimensional space (if necessary).
2. Find the optimal hyperplane using support vectors.
3. Predict based on which side of the hyperplane a point lies.

---

## Mathematics
### Optimization Problem
- Maximize the margin: $\frac{1}{\|w\|}$
- Subject to: $y_i(w \cdot x_i + b) \geq 1$

### Dual Problem (Kernelized SVM)
- $\max \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$
- Subject to: $\sum \alpha_i y_i = 0, \; 0 \leq \alpha_i \leq C$

### Decision Function
- $f(x) = \text{sign}(\sum \alpha_i y_i K(x_i, x) + b)$

---

## Code
### Classification
```python
from sklearn.svm import SVC
import numpy as np

# Sample data
X = np.array([[1, 2], [2, 3], [3, 3], [4, 5]])
y = np.array([1, 1, 0, 0])

# Model
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# Prediction
query = np.array([[2.5, 3.0]])
print(clf.predict(query))
```

### Non-Linear Classification
```python
# Radial Basis Function (RBF) kernel
clf = SVC(kernel='rbf', C=1.0, gamma=0.5)
clf.fit(X, y)
print(clf.predict(query))
```

### Regression
```python
from sklearn.svm import SVR

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([1.5, 2.0, 3.5, 4.0])

# Model
reg = SVR(kernel='linear', C=1.0)
reg.fit(X, y)

# Prediction
query = np.array([[2.5]])
print(reg.predict(query))
```

---

## Key Points to Remember
- **Hyperparameters**:
  - $C$: Regularization parameter (controls margin width and misclassification tolerance).
  - Kernel: Determines the decision boundary shape (linear, polynomial, RBF, etc.).
  - $\gamma$: Controls influence of points in RBF kernel.
- **Support Vectors**: Only a subset of training points (support vectors) define the decision boundary.

---

## Extensions
- **Kernel SVM**: Handles non-linear data with kernels.
- **One-Class SVM**: For outlier detection.
- **SVM with SGD**: Stochastic Gradient Descent implementation for scalability.

---

## Pros
- Effective in high-dimensional spaces.
- Works well with a clear margin of separation.
- Can handle non-linear data with appropriate kernels.

---

## Cons
- Computationally expensive for large datasets.
- Sensitive to the choice of hyperparameters.
- Struggles with noisy or overlapping classes.

---

## Applications
- Text classification (e.g., spam detection).
- Image recognition.
- Bioinformatics (e.g., protein classification).

---

## Limitations
- Requires feature scaling (e.g., normalization).
- Poor performance with imbalanced datasets.
- Does not provide direct probability estimates (can be approximated with Platt scaling).

---