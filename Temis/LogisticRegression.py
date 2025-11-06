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

    def _cost_function(self, params, X : np.ndarray, y : np.ndarray, S : np.ndarray = None, debug : bool = False) -> float:
        w, b = params
        m = X.shape[0]
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        if S is not None:
            S = jnp.asarray(S)
        print(type(w))
        print(type(b))

        z = jnp.dot(X, w) + b
        y_pred = sigmoid(z)
        
        eps = 1e-5
        base_cost = -jnp.mean(y * y_pred + (1-y) * (1 - y_pred))

        reg_cost = 0.0
        if self.penalty == 'l1':
            reg_cost = (self.penalty_weight / 2) * jnp.sum(jnp.abs(w))
        elif self.penalty == 'l2':
            reg_cost = (self.penalty_weight / 2) * jnp.sum(w ** 2)

        fair_cost = 0.0
        if self.fair_penalty == 'Rpr':
            M = y * y_pred + (1-y) * (1 - y_pred)
            pr_hat_y = jnp.mean(M) + eps

            mask0 = (S == 0)
            mask1 = (S == 1)
            count0 = jnp.sum(mask0)
            count1 = jnp.sum(mask1)
            div0 = jnp.max(count0, 1)
            div1 = jnp.max(count1, 1)

            pr_hat_y_given_s0 = jnp.sum(M * mask0) / div0
            pr_hat_y_given_s1 = jnp.sum(M * mask1) / div1

            ln_ratio_given_s0 = jnp.log(pr_hat_y_given_s0 / pr_hat_y + eps)
            ln_ratio_given_s1 = jnp.log(pr_hat_y_given_s1 / pr_hat_y + eps)
            #print(f'ln_ratio_given_s0: {ln_ratio_given_s0}')
            #print(f'ln_ratio_given_s1: {ln_ratio_given_s1}')

            ln_ratio = ln_ratio_given_s0 + ln_ratio_given_s1
            fair_cost = jnp.sum(M * ln_ratio) 
        #if self.fair_penalty == 'Rpr':
            #if S is None:
                #raise ValueError("Sensitive attribute S must be provided for fairness penalty.")
            #p_y1 = jnp.mean(y_pred)
            #p_y0 = jnp.mean(y_pred_0)

            #if(debug == True):
                #print(f"p_y1: {p_y1}, p_y0: {p_y0}")

            #mask0 = (S == 0)
            #mask1 = (S == 1)
            #count0 = jnp.sum(mask0)
            #count1 = jnp.sum(mask1)
            #div0 = jnp.maximum(count0, 1)
            #div1 = jnp.maximum(count1, 1)
            
            #if(debug == True):
                #print(f'div0: {div0}, div1: {div1}')

            #p_y0_s0 = jnp.sum(y_pred_0 * mask0) / div0
            #p_y0_s1 = jnp.sum(y_pred_0 * mask1) / div1
            #p_y1_s0 = jnp.sum(y_pred * mask0) / div0
            #p_y1_s1 = jnp.sum(y_pred * mask1) / div1

            #if(debug == True):
                #print(f"p_y0_s0: {p_y0_s0}, p_y0_s1: {p_y0_s1}, p_y1_s0: {p_y1_s0}, p_y1_s1: {p_y1_s1}")

            #ratio_y0_s0 = p_y0_s0 / (p_y0 + eps)
            #ratio_y0_s1 = p_y0_s1 / (p_y0 + eps)
            #ratio_y1_s0 = p_y1_s0 / (p_y1 + eps)
            #ratio_y1_s1 = p_y1_s1 / (p_y1 + eps)

            #if(debug == True):
                #print(f"ratio_y0_s0: {ratio_y0_s0}, ratio_y0_s1: {ratio_y0_s1}, ratio_y1_s0: {ratio_y1_s0}, ratio_y1_s1: {ratio_y1_s1}")

            #log_ratio_y0 = jnp.where(mask0, jnp.log(ratio_y0_s0 + eps), jnp.log(ratio_y0_s1 + eps))
            #log_ratio_y1 = jnp.where(mask0, jnp.log(ratio_y1_s0 + eps), jnp.log(ratio_y1_s1 + eps))
            
            #if(debug == True):
                #print(f"log_ratio_y0: {log_ratio_y0}, log_ratio_y1: {log_ratio_y1}")

            #rpr = jnp.mean(y_pred * log_ratio_y1 + y_pred_0 * log_ratio_y0)
            #fair_cost = self.fair_penalty_weight * rpr
        
        #if debug == True:
            #print(f"Base Cost: {base_cost}, Reg Cost: {reg_cost}, Fair Cost: {fair_cost}")
        cost = base_cost + reg_cost + fair_cost
        return cost

    def fit(self, X : np.ndarray, y : np.ndarray, S : np.ndarray = None, debug : bool = False):
        if(debug == True):
            print('----------------------------------------------------------------------')
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        S = jnp.asarray(S) if S is not None else None

        m, n =  X.shape
        self.w = jnp.zeros(n)
        self.b = 0.0

        # add jit
        cost_grad = (jax.grad(self._cost_function, argnums=0))

        for epoch in range(self.epochs):
            perm = np.random.permutation(m)
            X_shuffle = X[perm]
            y_shuffle = y[perm]
            S_shuffle = S[perm] if S is not None else None

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffle[i:i+self.batch_size]
                y_batch = y_shuffle[i:i+self.batch_size]
                S_batch = S_shuffle[i:i+self.batch_size] if S is not None else None

                grads = cost_grad((self.w, self.b), X_batch, y_batch, S_batch)
                dw, db = grads

                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            if debug == True:
                print("Debug Info:")
                cost = self._cost_function((self.w, self.b), X, y, S, debug = True)
                print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost}")
                print(f"w: {self.w}, b: {self.b}")


    def predict_probability(self, X : np.ndarray) -> np.ndarray:
        X = jnp.asarray(X)
        return sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        X = jnp.asarray(X)
        return (self.predict_probability(X) >= threshold).astype(int)

    def print_debug_statement(self):
        print('This change must appear in the notebook')
        print('Testing reload functionality.')
