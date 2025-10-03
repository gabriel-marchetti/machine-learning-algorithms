import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Temis.LogisticRegression import LogisticRegression

if __name__ == "__main__":
    n_features = 5
    X, y = make_classification(
        n_samples=1000,
        n_features=n_features,
        n_informative=3,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = 0.1
    epochs = 100
    model = LogisticRegression(lr, epochs)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    acc_train = jnp.mean(pred_train == y_train)
    acc_test = jnp.mean(pred_test == y_test)

    print(f"Train Accuracy: {acc_train:.2f}")
    print(f"Test Accuracy: {acc_test:.2f}")

    fig, axes = plt.subplots(n_features, n_features, figsize=(12, 12))
    fig.suptitle(f'Fronteira de decisão para cada feature Par a Par | Accuracy : {acc_test:.2f}', fontsize = 20)
    X_mean = X_test.mean(axis=0)

    miss = (y_test != pred_test)

    for i in range(n_features):
        for j in range(n_features):
            if(i == j):
                continue

            ax = axes[i, j]

            ax.scatter(
                X_test[:, j],
                X_test[:, i],
                c=y_test,
                cmap=plt.cm.coolwarm,
                s = 25,
                edgecolor="k",
                alpha=0.7
            )

            ax.scatter(
                X_test[miss, j],
                X_test[miss, i],
                c='yellow',
                s = 25,
                edgecolor='k',
                label='Miss'
            )

            x_min, x_max = X_test[:, j].min() - 1, X_test[:, j].max() + 1
            y_min, y_max = X_test[:, i].min() - 1, X_test[:, i].max() + 1
            
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100), 
                np.linspace(y_min, y_max, 100)
            )
            
            grid_points_full = np.tile(X_mean, (xx.ravel().shape[0], 1))
            grid_points_full[:, j] = xx.ravel()
            grid_points_full[:, i] = yy.ravel()
            
            Z = model.predict(grid_points_full)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    plt.show()