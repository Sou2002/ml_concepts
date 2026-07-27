# transparentml

Transparent ML is a lightweight, educational machine learning library built from scratch with NumPy. It provides simple implementations of common supervised and unsupervised learning algorithms for experimentation, learning, and small projects.

## Why choose transparentml?

- Simple, readable implementations designed for learning and teaching.
- Built with NumPy only, with no heavy dependencies.
- Includes both classical ML algorithms and basic clustering methods.
- Easy to inspect and modify for educational purposes.

## Features

The package currently includes:

- Linear regression
- Logistic regression
- SGD regression
- K-nearest neighbors
- K-means clustering
- Decision trees

## Installation

Install from PyPI:

```bash
pip install transparentml
```

## Quick start

```python
import numpy as np
from transparentml.linear_models.linear_regression import LinearRegression

X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(np.array([[5.0]]))
print(predictions)
```

## Example usage

### Linear regression

```python
from transparentml.linear_models.linear_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Logistic regression

```python
from transparentml.linear_models.logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.1, epochs=100)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```

### K-means clustering

```python
from transparentml.clustering.kmeans import KMeans

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)
```

### K-nearest neighbors

```python
from transparentml.neighbors.knn import KNN

model = KNN(n_neighbours=5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Decision tree

```python
from transparentml.tree.decision_tree import DecisionTree

model = DecisionTree(min_samples_split=2, max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Project status

This project is currently in early development and is best suited for learning, experimentation, and educational use. It is not intended to be a full replacement for production-grade libraries such as scikit-learn.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome. If you want to improve the package, add examples, or fix issues, feel free to open a pull request or issue on GitHub.
