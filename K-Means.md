# K-Means Clustering

## What
- **K-Means** is an unsupervised learning algorithm used for clustering data into $K$ distinct groups.
- It minimizes the variance within clusters by iteratively refining cluster centroids.

## Why
- To find natural groupings in data.
- Useful for exploratory data analysis, pattern recognition, and dimensionality reduction.

## How
1. Choose the number of clusters $K$.
2. Initialize $K$ centroids randomly or with a heuristic (e.g., K-Means++).
3. Repeat until convergence:
   - Assign each data point to the nearest centroid.
   - Update centroids as the mean of assigned points.
4. Stop when centroids no longer change or after a maximum number of iterations.

---

## Assumptions
- Data points are grouped around centroids.
- Euclidean distance is appropriate for measuring similarity.
- Clusters are roughly spherical and equally sized.

---

## Example

### Intuition
- Group customers into $K$ clusters based on their purchasing behavior.

### Data
```plaintext
| Feature 1 | Feature 2 |
|-----------|-----------|
| 1.0       | 1.5       |
| 1.2       | 1.8       |
| 3.5       | 3.0       |
| 3.8       | 3.2       |
```

### Process
1. Initialize two centroids (e.g., $(1,1)$ and $(3,3)$).
2. Assign points to the closest centroid.
3. Update centroids based on assigned points.
4. Repeat until centroids stabilize.

---

## Mathematics
### Objective Function
- Minimize within-cluster variance:
  $$J = \sum_{i=1}^K \sum_{x \in C_i} \|x - \mu_i\|^2$$
  - $K$: Number of clusters.
  - $C_i$: Cluster $i$.
  - $\mu_i$: Centroid of cluster $i$.

### Update Rules
- Assign clusters: $c_i = \arg\min_k \|x_i - \mu_k\|^2$
- Update centroids: $\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$

---

## Code
```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1.0, 1.5], [1.2, 1.8], [3.5, 3.0], [3.8, 3.2]])

# Model
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print("Labels:", labels)
print("Centroids:", centroids)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='x')
plt.show()
```

---

## Key Points to Remember
- **Initialization Matters**: Poor initialization can lead to suboptimal solutions (use K-Means++ for better initialization).
- **Convergence**: Algorithm stops when centroids stabilize or max iterations are reached.
- **Distance Metric**: Assumes Euclidean distance; alternative metrics require modifications.

---

## Extensions
- **K-Means++**: Better initialization for faster convergence.
- **Mini-Batch K-Means**: Scalable version for large datasets.
- **Kernel K-Means**: Uses kernel functions for non-linear separations.

---

## Pros
- Simple and easy to implement.
- Scalable to large datasets.
- Works well when clusters are distinct and spherical.

---

## Cons
- Sensitive to outliers and noise.
- Requires the number of clusters ($K$) to be specified in advance.
- Struggles with non-spherical or overlapping clusters.

---

## Applications
- Customer segmentation.
- Image compression.
- Anomaly detection.
- Document clustering.

---

## Limitations
- Struggles with high-dimensional data.
- Not deterministic (results vary due to random initialization).
- Assumes clusters are of similar size and density.

---