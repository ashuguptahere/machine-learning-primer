# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## What
- **DBSCAN**: A density-based clustering algorithm that groups points closely packed together, marking outliers as noise.
- Detects arbitrarily shaped clusters and separates noise effectively.

## Why
- To identify clusters of varying shapes and densities in a dataset.
- Handles noise and outliers better than centroid-based methods like k-means.
- Works without requiring the number of clusters beforehand.

## How
1. **Core Points**: Points with at least `min_samples` neighbors within a radius `eps`.
2. **Border Points**: Points within `eps` of a core point but with fewer than `min_samples` neighbors.
3. **Noise**: Points not reachable from any core point.
4. Clusters are formed by connecting core points and their reachable neighbors.

## When (to use)
- Data has noise and outliers.
- Clusters vary in shape and size.
- No prior knowledge of the number of clusters.
- Distance-based similarity measure makes sense.

## Assumptions
- Density can define clusters.
- All points within a cluster are reachable from one another.
- Requires a meaningful distance metric.

## Example

**Dataset**
| Point | X   | Y   |
|-------|-----|-----|
| A     | 1.0 | 1.1 |
| B     | 1.2 | 1.0 |
| C     | 5.0 | 5.1 |
| D     | 5.2 | 5.0 |
| E     | 8.0 | 8.0 |
| F     | 1.1 | 1.2 |

**Intuition**
- `eps = 0.5`, `min_samples = 3`
- A, B, F: Core points (dense region)
- C, D: Border points (reachable from a core point)
- E: Noise (isolated)

## Mathematics
- **Core Point Criterion**: 
  $$\text{Neighbors}(p) = \{q \mid \text{distance}(p, q) \leq \epsilon\}$$
  $$\text{If } |\text{Neighbors}(p)| \geq \text{min_samples}, \text{then } p \text{ is a core point.}$$

- **Distance Function**:
  $$\text{distance}(x_i, x_j) = \sqrt{\sum_{k=1}^n (x_{ik} - x_{jk})^2}$$ (Euclidean distance)

## Code
```python
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Sample Data
X = np.array([
    [1.0, 1.1], [1.2, 1.0], [5.0, 5.1],
    [5.2, 5.0], [8.0, 8.0], [1.1, 1.2]
])

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=3)
labels = db.fit_predict(X)

# Plot Results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

## Key Points to Remember
- `eps`: Defines the neighborhood radius.
- `min_samples`: Minimum points required to form a dense region.
- Handles noise by labeling points as `-1`.
- Results depend on the choice of `eps` and `min_samples`.

## Extensions
- **OPTICS**: Extends DBSCAN for variable density.
- **HDBSCAN**: Hierarchical extension for varying densities and hierarchical clustering.

## Pros
- Detects arbitrarily shaped clusters.
- Does not require the number of clusters as input.
- Handles noise effectively.

## Cons
- Sensitive to `eps` and `min_samples` parameters.
- Struggles with varying densities.
- Computationally expensive for large datasets.

## Applications
- Geographic data clustering.
- Anomaly detection in network traffic.
- Image segmentation.
- Social network analysis.

## Limitations
- Difficulty in choosing optimal `eps` and `min_samples`.
- Performance degrades with high-dimensional data.
- Requires a well-defined distance metric.

---