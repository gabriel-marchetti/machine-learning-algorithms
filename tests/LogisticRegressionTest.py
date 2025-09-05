import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from LogisticRegression import LogisticRegression

if __name__ == "__main__":
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42
    )

    lr = 0.1
    epochs = 100
    model = LogisticRegression(lr, epochs)
    model.fit(X, y)

    pred = model.predict(X)
    acc = np.mean(pred == y)

    print(f"Accuracy: {acc:.2f}")

    # Creating Decision Boundary Plot:
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        s=40,
        edgecolor="k"
    )

    misclassified = y != pred
    plt.scatter(
        X[misclassified, 0],
        X[misclassified, 1],
        c="yellow",
        s=60,
        edgecolor="k",
        label="Classificação Errada"
    )

    plt.title(f"Logistic Regression | Accuracy = {acc:.2f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()