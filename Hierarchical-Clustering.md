# Hierarchical Clustering

## What
- **Hierarchical Clustering**: A clustering technique that builds a hierarchy of clusters.
- Two main approaches:
  - **Agglomerative**: Bottom-up approach; starts with each point as its own cluster, merges clusters iteratively.
  - **Divisive**: Top-down approach; starts with all points in one cluster, splits clusters iteratively.

## Why
- Understand hierarchical relationships in data.
- No need to specify the number of clusters beforehand.
- Suitable for data with inherent nested structures.

## How
1. Compute pairwise distances between points.
2. Merge or split clusters based on a linkage criterion.
3. Repeat until desired cluster structure is achieved.
4. Visualize results using a **dendrogram**.

### Linkage Criteria
- **Single Linkage**: Minimum distance between points of two clusters.
- **Complete Linkage**: Maximum distance between points of two clusters.
- **Average Linkage**: Average distance between points of two clusters.
- **Ward’s Method**: Minimizes variance within clusters.

## When (to use)
- Small to medium-sized datasets.
- Data with a nested or hierarchical structure.
- Visual exploration of clustering using dendrograms.

## Assumptions
- Data points can be grouped into a hierarchy.
- A meaningful distance metric exists.

## Example

**Dataset**
| Point | X   | Y   |
|-------|-----|-----|
| A     | 1.0 | 1.1 |
| B     | 1.2 | 1.0 |
| C     | 5.0 | 5.1 |
| D     | 5.2 | 5.0 |
| E     | 8.0 | 8.0 |

**Intuition**
- Calculate distances between points (e.g., Euclidean).
- Start with each point as its own cluster.
- Merge closest clusters iteratively based on chosen linkage.

## Mathematics
- **Distance Matrix**:
  $$\text{distance}(x_i, x_j) = \sqrt{\sum_{k=1}^n (x_{ik} - x_{jk})^2}$$

- **Linkage** (Example: Average Linkage):
  $$\text{distance}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{p \in C_i} \sum_{q \in C_j} \text{distance}(p, q)$$

## Code
```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Sample Data
X = np.array([
    [1.0, 1.1], [1.2, 1.0], [5.0, 5.1],
    [5.2, 5.0], [8.0, 8.0]
])

# Perform Hierarchical Clustering
Z = linkage(X, method='ward')

# Plot Dendrogram
dendrogram(Z, labels=['A', 'B', 'C', 'D', 'E'])
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()
```

## Key Points to Remember
- **Agglomerative clustering** is more common than divisive.
- Dendrogram helps visualize the hierarchy and decide the number of clusters.
- Linkage method affects the cluster structure significantly.
- Computationally expensive for large datasets.

## Extensions
- **BIRCH**: Efficient hierarchical clustering for large datasets.
- **CUTREE**: Cutting a dendrogram to form clusters.

## Pros
- No need to predefine the number of clusters.
- Produces a full hierarchy of clusters.
- Can capture nested cluster structures.

## Cons
- Computationally intensive for large datasets.
- Sensitive to distance metric and linkage method.
- Difficult to decide the optimal number of clusters from a dendrogram.

## Applications
- Taxonomy (biological classification).
- Document clustering.
- Image segmentation.
- Social network analysis.

## Limitations
- High memory and computational cost for large datasets.
- Requires careful selection of distance metric and linkage method.
- Does not perform well on data with overlapping clusters.

---