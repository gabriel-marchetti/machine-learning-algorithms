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
                 penalty : str = None,
                 penalty_weight : float = 0.01,
                 fair_penalty : str = None,
                 fair_penalty_weight : float = 0.01):
        '''
        Class Contructor.
        Parameters:
            lr (float): Learning rate for gradient descent.     (Default is 0.1)
            epochs (int): Number of epochs for training.        (Default is 100)
            batch_size (int): Size of mini-batches for training.(Default is 32)
            penalty (str): Type of regularization to apply. 
                Options are 'l2', 'l1', or None.                (Default is None)
            penalty_weight (float): Regularization strength.        (Default is 0.01)
            fair_penalty (str): Type of fairness regularization to apply.
                Options are 'Rpr' or None.                      (Default is None)
            fair_penalty_weight (float): Fairness regularization strength. 
                                                                (Default is 0.01)
        '''
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.penalty_weight = penalty_weight
        self.fair_penalty = fair_penalty
        self.fair_penalty_weight = fair_penalty_weight
        self.w = None
        self.b = None

    def _cost_function(self, params, X : np.ndarray, y : np.ndarray, S : np.ndarray = None) -> float:
        w, b = params
        m = X.shape[0]

        z = jnp.dot(X, w) + b
        y_pred = sigmoid(z)
        
        eps = 1e-9
        base_cost = -jnp.mean(y * jnp.log(y_pred + eps) + (1-y) * jnp.log(1 - y_pred + eps))

        reg_cost = 0.0
        if self.penalty == 'l1':
            reg_cost = self.penalty_weight * jnp.sum(jnp.abs(w))
        elif self.penalty == 'l2':
            reg_cost = self.penalty_weight * jnp.sum(w ** 2)

        fair_cost = 0.0

        if self.fair_penalty == 'Rpr':
            p_y1 = jnp.mean(y_pred) + eps 
            p_y0 = 1 - p_y1 + eps

            p_y0_s0 = jnp.mean(1 - y_pred[S == 0])
            p_y1_s1 = jnp.mean(y_pred[S == 1])

            ratio0 = jnp.clip(p_y0_s0 / p_y0, eps, 1/eps)
            ratio1 = jnp.clip(p_y1_s1 / p_y1, eps, 1/eps)

            fair_cost = self.fair_penalty_weight * (jnp.sum((1-y_pred) * ratio0 + (-y_pred) * ratio1))

        cost = base_cost + reg_cost + fair_cost
        return cost

    def fit(self, X : np.ndarray, y : np.ndarray, S : np.ndarray = None, debug : bool = False):
        m, n =  X.shape
        self.w = jnp.zeros(n)
        self.b = 0.0

        grad = jax.grad(self._cost_function, argnums=0)

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)
            X_shuffle = X[perm]
            y_shuffle = y[perm]
            S_shuffle = S[perm] if S is not None else None

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffle[i:i+self.batch_size]
                y_batch = y_shuffle[i:i+self.batch_size]
                S_batch = S_shuffle[i:i+self.batch_size] if S is not None else None

                grads = grad((self.w, self.b), X_batch, y_batch, S_batch)
                dw, db = grads

                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            if debug == True:
                cost = self._cost_function((self.w, self.b), X, y)
                print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost}")
                print(f"w: {self.w}, b: {self.b}")


    def predict_probability(self, X : np.ndarray) -> np.ndarray:
        return sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        return (self.predict_probability(X) >= threshold).astype(int)

    def print_debug_statement(self):
        print('This change must appear in the notebook')
        print('Testing reload functionality.')
