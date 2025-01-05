# Types of Machine Learning

## 1. Supervised Learning
- **Definition**: Learning from labeled data where the output is known.
- **Purpose**:
  - Predict outputs for new, unseen data.
  - Build models that map inputs to outputs.
- **Examples**:
  - Classification (e.g., email spam detection).
  - Regression (e.g., house price prediction).
- **Workflow**:
  1. Provide labeled training data: $(X, y)$.
  2. Train the model to minimize error between predicted and actual output.
  3. Validate the model on test data.
- **Key Algorithms**:
  - Linear Regression, Logistic Regression, SVM, Decision Trees, Neural Networks.

---

## 2. Semi-Supervised Learning
- **Definition**: Learning from a mix of labeled and unlabeled data.
- **Purpose**:
  - Utilize unlabeled data to improve model accuracy.
  - Reduce the dependency on labeled data.
- **Examples**:
  - Text classification with limited labeled documents.
  - Image recognition with partially labeled datasets.
- **Workflow**:
  1. Train the model on labeled data.
  2. Use the model to label the unlabeled data.
  3. Retrain using the expanded labeled dataset.
- **Key Algorithms**:
  - Self-training, Label Propagation, Semi-Supervised SVM.

---

## 3. Unsupervised Learning
- **Definition**: Learning from data without labeled outputs.
- **Purpose**:
  - Discover hidden patterns or structures in data.
  - Explore data for clustering or dimensionality reduction.
- **Examples**:
  - Clustering (e.g., customer segmentation).
  - Dimensionality Reduction (e.g., PCA for visualization).
- **Workflow**:
  1. Provide data $X$ without labels.
  2. Use algorithms to find structures or groupings.
- **Key Algorithms**:
  - K-Means, DBSCAN, Hierarchical Clustering, PCA, Autoencoders.

---

## 4. Reinforcement Learning
- **Definition**: Learning by interacting with an environment to maximize cumulative reward.
- **Purpose**:
  - Solve sequential decision-making problems.
  - Learn optimal actions through trial and error.
- **Examples**:
  - Game AI (e.g., AlphaGo).
  - Robotics (e.g., path planning).
- **Workflow**:
  1. Agent takes action $a_t$ based on state $s_t$.
  2. Environment provides reward $r_t$ and new state $s_{t+1}$.
  3. Update policy to maximize total reward.
- **Key Concepts**:
  - **Policy**: Strategy to determine actions.
  - **Reward**: Feedback from the environment.
  - **Value Function**: Expected cumulative reward from a state.
  - **Q-Learning**: Learn action-value pairs $Q(s, a)$.
- **Key Algorithms**:
  - Q-Learning, Deep Q-Networks (DQN), Policy Gradient Methods, Actor-Critic.

---

## Comparisons
| **Type**             | **Labeled Data**         | **Use Cases**                           | **Examples**                           |
|-----------------------|--------------------------|-----------------------------------------|-----------------------------------------|
| Supervised Learning  | Fully labeled            | Predict outcomes                       | Email filtering, regression problems   |
| Semi-Supervised      | Partially labeled        | Improve with limited labels            | Text and image classification          |
| Unsupervised         | No labels                | Discover patterns                      | Customer segmentation, PCA             |
| Reinforcement        | Rewards and penalties    | Sequential decision making             | Robotics, game playing                 |

---