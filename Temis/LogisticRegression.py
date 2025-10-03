import numpy as np
import jax
import jax.numpy as jnp
from Temis.math_utils._functions import sigmoid

"""
    
"""
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=100, batch_size = 32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _cost_function(self, params, X : np.ndarray, y : np.ndarray):
        w, b = params
        m = X.shape[0]

        z = jnp.dot(X, w) + b
        y_pred = sigmoid(z)
        
        eps = 1e-9
        cost = -jnp.mean(y * jnp.log(y_pred + eps) + (1-y) * jnp.log(1 - y_pred + eps))
        return cost

    def fit(self, X : np.ndarray, y : np.ndarray):
        m, n =  X.shape
        self.w = jnp.zeros(n)
        self.b = 0.0

        grad = jax.grad(self._cost_function, argnums=0)

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)
            X_shuffle = X[perm]
            y_shuffle = y[perm]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffle[i:i+self.batch_size]
                y_batch = y_shuffle[i:i+self.batch_size]

                grads = grad((self.w, self.b), X_batch, y_batch)
                dw, db = grads

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict_probability(self, X : np.ndarray) -> np.ndarray:
        return sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        return (self.predict_probability(X) >= threshold).astype(int)
