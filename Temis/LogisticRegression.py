import numpy as np
import jax
import jax.numpy as jnp
from Temis.math_utils._functions import sigmoid

"""
    Class that implements Logistic Regression using JAX for automatic differentiation and optimization.    
    Description:
        
"""
class LogisticRegression:
    def __init__(self,
                 lr=0.1, 
                 epochs=100, 
                 batch_size = 32,
                 regularization : str = None,
                 lambda_reg : float = 0.01,
                 fairness_regularization : str = None,
                 lambda_fairness : float = 0.01):
        '''
        Class Contructor.
        Parameters:
            lr (float): Learning rate for gradient descent.     (Default is 0.1)
            epochs (int): Number of epochs for training.        (Default is 100)
            batch_size (int): Size of mini-batches for training.(Default is 32)
            regularization (str): Type of regularization to apply. 
                Options are 'l2', 'l1', or None.                (Default is None)
            lambda_reg (float): Regularization strength.        (Default is 0.01)
            fairness_regularization (str): Type of fairness regularization to apply.
                Options are 'Rpr' or None.                      (Default is None)
            lambda_fairness (float): Fairness regularization strength. 
                                                                (Default is 0.01)
        '''
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.fairness_regularization = fairness_regularization
        self.lambda_fairness = lambda_fairness
        self.w = None
        self.b = None

    def _cost_function(self, params, X : np.ndarray, y : np.ndarray):
        w, b = params
        m = X.shape[0]

        z = jnp.dot(X, w) + b
        y_pred = sigmoid(z)
        
        eps = 1e-9
        base_cost = -jnp.mean(y * jnp.log(y_pred + eps) + (1-y) * jnp.log(1 - y_pred + eps))

        reg_cost = 0.0
        if self.regularization == 'l1':
            reg_cost = self.lambda_reg * jnp.sum(jnp.abs(w))
        elif self.regularization == 'l2':
            reg_cost = (self.lambda_reg / 2) * jnp.sum(w ** 2)

        fair_cost = 0.0
#        if self.fairness_regularization == 'Rpr':
#            p_y1 = y_pred 
#            p_y0 = 1 - y_pred
#
#            p_hat_y1 = jnp.mean(p_y1)
#            p_hat_y0 = jnp.mean(p_y0)
#
#            p_hat_y0_given_s1 = 

        cost = base_cost + reg_cost + fair_cost
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
