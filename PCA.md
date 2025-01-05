# PCA (Principal Component Analysis)

## What
- **PCA**: A dimensionality reduction technique that transforms data into a set of orthogonal components.
- Captures the directions (principal components) with the maximum variance in the data.
- Reduces the number of features while preserving important patterns.

## Why
- Simplifies data visualization and analysis by reducing dimensions.
- Removes noise and redundant features.
- Speeds up computations for high-dimensional data.

## How
1. Standardize the dataset (mean = 0, variance = 1).
2. Compute the covariance matrix of the data.
3. Calculate the eigenvalues and eigenvectors of the covariance matrix.
4. Sort eigenvectors by their eigenvalues in descending order.
5. Select the top `k` eigenvectors to form the projection matrix.
6. Transform the data using the projection matrix.

## When (to use)
- Data has many correlated features.
- Visualization of high-dimensional data is needed.
- Preprocessing for machine learning to reduce overfitting.

## Assumptions
- Data is linear, and most variance can be captured in fewer dimensions.
- Features are continuous and numerical.

## Example

**Dataset**
| Sample | Feature 1 | Feature 2 |
|--------|-----------|-----------|
| A      | 2.5       | 2.4       |
| B      | 0.5       | 0.7       |
| C      | 2.2       | 2.9       |
| D      | 1.9       | 2.2       |
| E      | 3.1       | 3.0       |

**Intuition**
- Data is distributed along two correlated features.
- PCA finds new axes (principal components) that maximize variance.
- Example: First principal component might align with the direction of maximum spread.

## Mathematics
1. **Standardization**:
   $$z = \frac{x - \mu}{\sigma}$$

2. **Covariance Matrix**:
   $$\text{Cov}(X) = \frac{1}{n-1} (X^T X)$$

3. **Eigenvalue Decomposition**:
   $$\text{Cov}(X) v = \lambda v$$
   - $\lambda$: Eigenvalues (variance explained by components).
   - $v$: Eigenvectors (principal components).

4. **Projection**:
   $$Z = X W_k$$
   - $W_k$: Top `k` eigenvectors.

## Code
```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample Data
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0]
])

# PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Variance Explained
explained_variance = pca.explained_variance_ratio_

# Plot Original vs Transformed
plt.scatter(X[:, 0], X[:, 1], label="Original Data", alpha=0.7)
plt.scatter(X_pca, [0]*len(X_pca), label="PCA (1 Component)", alpha=0.7)
plt.title("PCA Transformation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
```

## Key Points to Remember
- PCA components are orthogonal.
- The first principal component captures the most variance.
- Standardization is crucial if features have different scales.
- Explained variance ratio indicates the importance of each component.

## Extensions
- **Kernel PCA**: Handles non-linear data by applying kernel functions.
- **Sparse PCA**: Incorporates sparsity to interpret principal components.

## Pros
- Reduces dimensionality effectively.
- Handles correlated features well.
- Improves interpretability of data.

## Cons
- Assumes linearity.
- Sensitive to outliers.
- Components may lose interpretability.

## Applications
- Image compression.
- Visualization of high-dimensional data.
- Feature extraction for machine learning.
- Genomic data analysis.

## Limitations
- Cannot capture non-linear relationships.
- Sensitive to the scale of features.
- Does not work well if variance does not correlate with importance.

---