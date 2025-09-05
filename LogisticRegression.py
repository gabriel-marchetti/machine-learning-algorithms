import numpy as np
from math_utils._functions import sigmoid

"""
    
"""
class LogisticRegression:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X : np.ndarray, y : np.ndarray):
        m, n =  X.shape
        self.w = np.zeros(n)
        self.b = 0.0

        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b 
            y_pred = sigmoid(z)

            dw  = (1/m) * np.dot(X.T, (y_pred - y))
            db  = (1/m) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_probability(self, X : np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        return (self.predict_probability(X) >= threshold).astype(int)
