# K-Nearest Neighbors (KNN)

## What
- **K-Nearest Neighbors (KNN)** is a non-parametric, lazy learning algorithm used for classification and regression.
- It predicts the output based on the majority class (classification) or the average value (regression) of the **k nearest data points**.

## Why
- Simple and effective for low-dimensional data.
- Does not assume a functional form of the data distribution.

## How
- Choose **k** (number of neighbors).
- Measure the distance between the query point and all training points using a metric:
  - Common metrics: **Euclidean**, **Manhattan**, **Minkowski**.
- Identify the **k nearest neighbors**.
- For prediction:
  - **Classification**: Assign the majority class among neighbors.
  - **Regression**: Compute the average of the neighbor values.

---

## Assumptions
- Data points close to each other are similar.
- Relevant features are properly scaled (important for distance-based algorithms).

---

## Example

### Intuition
- Predict if a fruit is an apple or an orange based on **weight** and **size**.
- Nearest fruits to the query point determine the prediction.

### Data
```plaintext
| Weight | Size | Label    |
|--------|------|----------|
| 150    | 7.5  | Apple    |
| 180    | 8.0  | Apple    |
| 120    | 6.5  | Orange   |
```

### Process
1. Compute distance from query point to all training points.
2. Identify the closest **k** points.
3. Predict the output based on neighbor labels.

---

## Mathematics
### Distance Metrics
#### Euclidean Distance
- Formula: $d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$

#### Manhattan Distance
- Formula: $d = \sum_{i=1}^{n} |x_i - y_i|$

### Weighted KNN
- Assign weights to neighbors based on distance.
- Formula: $w_i = \frac{1}{d_i}$, where $d_i$ is the distance of the $i$-th neighbor.

---

## Code
### Classification
```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Sample data
X = np.array([[150, 7.5], [180, 8.0], [120, 6.5]])  # Weight, Size
y = np.array([0, 0, 1])  # Apple (0), Orange (1)

# Model
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)

# Prediction
query = np.array([[160, 7.8]])
print(knn.predict(query))
```

### Regression
```python
from sklearn.neighbors import KNeighborsRegressor

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Feature
y = np.array([5, 7, 9, 11, 13])          # Target

# Model
knn_reg = KNeighborsRegressor(n_neighbors=2, metric='euclidean')
knn_reg.fit(X, y)

# Prediction
query = np.array([[6]])
print(knn_reg.predict(query))
```

---

## Key Points to Remember
- **Choosing k**:
  - Small **k**: Sensitive to noise.
  - Large **k**: Over-smoothing.
- **Feature Scaling**: Normalize or standardize features.
- **Distance Metrics**: Euclidean is default; choose based on data.

---

## Extensions
- **Weighted KNN**: Weights neighbors by inverse distance.
- **Fast KNN**: Use KD-Tree or Ball-Tree for faster neighbor search.

---

## Pros
- Simple to implement.
- Effective for small datasets.
- No training phase (lazy learner).

---

## Cons
- Computationally expensive for large datasets.
- Sensitive to irrelevant or unscaled features.
- Requires storing the entire dataset in memory.

---

## Applications
- Recommendation systems.
- Image recognition.
- Customer segmentation.

---

## Limitations
- High computational cost for large datasets.
- Poor performance with imbalanced data.
- Struggles with high-dimensional data (curse of dimensionality).

---