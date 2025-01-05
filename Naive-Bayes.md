# Naive Bayes

## What
- **Naive Bayes**: A probabilistic machine learning algorithm based on Bayes' theorem.
- Assumes strong (naive) independence between features.
- Suitable for classification tasks.

---

## Why
- **Efficient**: Fast to train and predict.
- **Scalable**: Works well with large datasets.
- **Simple**: Easy to implement and interpret.
- Performs well for **text classification**, e.g., spam detection.

---

## How
1. Calculate prior probabilities for each class.
2. Compute the likelihood of each feature given the class.
3. Use Bayes' theorem to calculate the posterior probability for each class.
4. Choose the class with the highest posterior probability.

$$ P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} $$

Where:
- $P(C|X)$: Posterior probability of class $C$ given data $X$
- $P(X|C)$: Likelihood of data $X$ given class $C$
- $P(C)$: Prior probability of class $C$
- $P(X)$: Evidence (normalization factor)

---

## When (to use)
- When the dataset is **small** and **features are independent**.
- Effective for **text data** (e.g., bag-of-words representations).
- For **multi-class classification** problems.

---

## Assumptions
- **Feature Independence**: Features are conditionally independent given the class.
- **No Missing Data**: Assumes all features are present for every instance.
- Features are **categorical** or **convertible** to categorical (e.g., binning).

---

## Example
- **Task**: Classify emails as "spam" or "not spam."
- **Dataset**:
  - Email 1: "Free lottery tickets" → Spam
  - Email 2: "Meeting tomorrow" → Not Spam
  - Email 3: "Lottery winner" → Spam

- **Steps**:
  - Calculate the prior: $P(Spam) = \frac{2}{3}$, $P(Not\ Spam) = \frac{1}{3}$.
  - Compute likelihoods for words like "lottery," "tickets," etc.
  - Use Bayes' theorem to predict new emails based on word frequencies.

---

## Mathematics
1. **Bayes' Theorem**:
   $$ P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)} $$

2. **Naive Assumption**:
   $$ P(X|C) = P(x_1|C) \cdot P(x_2|C) \cdot \ldots \cdot P(x_n|C) $$

3. **Class Prediction**:
   $$ \hat{y} = \arg\max_C P(C|X) $$

---

## Code
```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample Data
texts = ["Free lottery tickets", "Meeting tomorrow", "Lottery winner"]
labels = [1, 0, 1]  # 1: Spam, 0: Not Spam

# Vectorize Text Data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

---

## Key Points to Remember
- Assumes **independence** among features.
- Works well with **categorical data**.
- Use **log probabilities** to avoid numerical underflow.
- Best suited for **text-based tasks**.

---

## Extensions
- **Bernoulli Naive Bayes**: For binary features.
- **Gaussian Naive Bayes**: For continuous data (assumes Gaussian distribution).
- **Complement Naive Bayes**: For imbalanced datasets.

---

## Pros
- **Fast** and computationally efficient.
- **Robust to irrelevant features**.
- Handles multi-class problems well.

---

## Cons
- Relies on **independence assumption** (rarely true).
- Poor performance when features are highly correlated.
- Requires **discretization** for continuous features.

---

## Applications
- **Spam detection**
- **Sentiment analysis**
- **Document classification**
- **Medical diagnosis**

---

## Limitations
- Fails with **highly correlated features**.
- Assumes **feature independence**, which is often unrealistic.
- Sensitive to **data imbalance**.

---