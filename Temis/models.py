import numpy as np
import jax
import jax.numpy as jnp
from sklearn.metrics import log_loss
from sklearn.base import clone
from Temis.math_utils._functions import sigmoid

'''
This Should implement the base class for MinimaxFairness method presented
in the paper https://arxiv.org/abs/2011.03108.

It works by implementing a Two-Player Game formulation of Learner and Regulator.
Learner: Will optimize objective function based on samples_weights and base cost function.
Regulator: Will adjust sample_weights for next turn of game and final implementation.


'''
class MinimaxFairness:
    '''
    Model structure:
        self.model_class : It is a class reference for the base method used.
        self.T : Iteration count on the number of games it will run.
        self.lr : adaptive learning rate, as shown in the paper it should be 1/sqrt(t) 
            where t denotes the current iteration of the game.
        self.n_groups : number of differente groups. (MAYBE ANOTHER NAME??)
        self.eps : OPT1 satisfatibility.
        self.debug : Enables debugging information.

        --- 
        Another useful information:
        self.models : holds all models that are produced in the game.
        self.lambdas_history : store the sample_weights history.
        self.group_losses_history : store the group_losses_history.
    '''
    def __init__(self, model_class, iter=1000, debug=False):
        self.model_class = model_class
        self.T = None
        self.lr = None
        self.n_groups = None
        self.iter = iter
        self.eps = None
        self.debug = debug

        # Initialize storage for models, lambdas, and group losses history
        self.models = []
        self.lambdas_history = []
        self.group_losses_history = []

    '''
    This method will implement the Two-Player game formulation logic.
    The game consists of a Learner and a Regulator that will play their turn.
    Learner turn: Optimize the objective function and return such parameters.
    Regulator turn: Will adjust sample_weights so that more important samples have greater weight.
    '''
    def fit(self, X, y, groups):
        if self.debug == True:
            print(f"[DEBUG] Debugging fit information...")
        n_samples = len(y)

        unique_groups = np.unique(groups)
        # This will be useful for defining iteration count self.T
        n_groups = len(unique_groups)
        self.n_groups = n_groups

        if self.debug == True:
            print(f"[DEBUG] Number identified groups: {self.n_groups}")

        # This bound is explicit defined in the paper.

        # Previous Design:
        # If i assume to use it in the future, self.K no longer exists and
        # now you must use self.n_groups
        # self.T = int(np.ceil(np.log(self.K) / (2 * self.eps * self.eps)))

        self.T = self.iter
        # This was defined in the paper
        self.eps = np.sqrt(np.log(n_groups)/(2*self.T))


        # Define proportions of each group.
        group_counts = {g: np.sum(groups == g) for g in unique_groups}

        if self.debug == True:
            print(f"[DEBUG] Group Counts: {group_counts}")

        # In MinimaxFair Algorithm, the first weight is defined by proportion of samples.
        self.lambdas = {g: group_counts[g] / n_samples for g in unique_groups}
        if self.debug == True:
            print(f"[DEBUG] Initial Lambdas: {self.lambdas}")

        # Game formulation...
        if self.debug == True:
            print(f"[DEBUG] Initializing game with {self.T} rounds...")
            
        #sample_weights = np.zeros(n_samples)
        #for g in unique_groups:
            #mask = (groups == g)
            #sample_weights[mask] = self.lambdas[g]

        for t in range(1, self.T + 1):
            self.lr = 1/np.sqrt(t)

            if self.debug == True:
                print(f"[DEBUG] Initialing round: {t}")

            sample_weights = np.zeros(n_samples)    
            for g in unique_groups:
                mask = (groups == g)
                sample_weights[mask] = self.lambdas[g]

            #h_t = self.model_class(solver='lbfgs', max_iter=100)
            h_t = clone(self.model_class)
            h_t.fit(X, y, sample_weight=sample_weights)
            self.models.append(h_t)

            group_losses = {}
            probs = h_t.predict_proba(X)

            for g in unique_groups:
                mask = (groups == g)
                loss_k = log_loss(y[mask], probs[mask])
                group_losses[g] = loss_k

            self.group_losses_history.append(group_losses)
            self.lambdas_history.append(self.lambdas.copy())

            for g in unique_groups:
                self.lambdas[g] *= np.exp(self.lr * group_losses[g])
    def predict_proba(self, X):
        if self.debug == True:
            print(f"[DEBUG] debugging predict_proba information....")
        preds = np.array([h.predict_proba(X) for h in self.models])
        mean_preds = np.mean(preds, axis=0)
        return mean_preds

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

"""
Class that implements Logistic Regression using JAX for automatic differentiation and optimization.    
Description:
"""
class LogisticRegression:
    def __init__(
            self,
            lr=0.1, 
            epochs=100, 
            batch_size = 32,
            penalty : str = None,
            penalty_weight : float = 0.01,
            fair_penalty : str = None,
            fair_penalty_weight : float = 0.01
        ):
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
        self.current_epoch = 0

    '''
    Private method to compute the cost function with optional regularization and fairness penalties.
    Parameters:
        params (tuple): Tuple containing weights and bias (w, b).
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        S (np.ndarray): Sensitive attribute for fairness penalty. (Default is None)
        debug (bool): Flag to enable debug output. (Default is False)
    Returns:
        float: Computed cost value.
    OBS:
        This function should only contain JAX operations to ensure compatibility with JAX automatic differentiation and
        JAX JIT compilation.
        The computation utilizes three components:
        1. Base Cost: Standard logistic regression loss.
        2. Regularization Cost: L1 or L2 regularization based on the specified penalty.
        3. Fairness Cost: Rpr fairness penalty if specified.
    '''
    def _cost_function(self, params, X : np.ndarray, y : np.ndarray, S : np.ndarray = None, debug : bool = False) -> float:
        w, b = params
        m = X.shape[0]
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        if S is not None:
            S = jnp.asarray(S)

        z = jnp.dot(X, w) + b
        y_pred_1 = sigmoid(z)
        y_pred_0 = 1 - y_pred_1
        
        eps = 1e-5
        base_cost = -jnp.mean(y * jnp.log(y_pred_1 + eps) + (1-y) * jnp.log(y_pred_0 + eps))

        l1_cost = self.penalty_weight * jnp.sum(jnp.abs(w))
        l2_cost = self.penalty_weight * jnp.sum(w ** 2)
        reg_cost = jnp.where(self.penalty == 'l1', l1_cost,
                             jnp.where(self.penalty == 'l2', l2_cost, 0.0))

        fair_cost = 0.0
        has_fairness = (self.fair_penalty == 'Rpr') & (S is not None)
        rpr = jnp.where(has_fairness,
                        compute_rpr(y_pred_1, y_pred_0, S, eps),
                        0.0)
        fair_cost = self.fair_penalty_weight * rpr
        
        cost = base_cost + reg_cost + fair_cost
        return cost

    '''
    Method to fit the Logistic Regression model to the data.
    Parameters:
        X (np.ndarray): Input features.
        y (np.ndarray): True labels.
        S (np.ndarray): Sensitive attribute for fairness penalty. (Default is None)
        debug (bool): Flag to enable debug output. (Default is False)
    Returns:
        None
    OBS:
        Uses JAX JIT and JAX Autodiff for efficient gradient computation and optimization.
        So be careful defining _cost_function to only use JAX operations.
    '''
    def fit(self, X : np.ndarray, y : np.ndarray, S : np.ndarray = None, debug : bool = False):
        if(debug == True):
            print('----------------------------------------------------------------------')
        X = jnp.asarray(X)
        y = jnp.asarray(y)
        S = jnp.asarray(S) if S is not None else None

        m, n =  X.shape
        self.w = jnp.zeros(n)
        self.b = 0.0

        cost_grad = jax.jit(jax.grad(self._cost_function, argnums=0))

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            perm = np.random.permutation(m)
            X_shuffle = X[perm]
            y_shuffle = y[perm]
            S_shuffle = S[perm] if S is not None else None

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffle[i:i+self.batch_size]
                y_batch = y_shuffle[i:i+self.batch_size]
                S_batch = S_shuffle[i:i+self.batch_size] if S is not None else None

                grads = cost_grad((self.w, self.b), X_batch, y_batch, S_batch, debug)
                dw, db = grads

                self.w -= self.lr * dw
                self.b -= self.lr * db

            if debug == True and self.current_epoch % 10 == 0:
                print(f'Epoch: {epoch}')
                print(f'Gradient Magnitude - dw: {jnp.mean(jnp.abs(dw))}')
                print(f'Gradient Magnitude - db: {jnp.abs(db)}')

    def predict_proba(self, X : np.ndarray) -> np.ndarray:
        X = jnp.asarray(X)
        return sigmoid(jnp.dot(X, self.w) + self.b)

    def predict(self, X : np.ndarray, threshold : float = 0.5) -> np.ndarray:
        X = jnp.asarray(X)
        return (self.predict_proba(X) >= threshold).astype(int)

    '''
    Method to print a debug statement to verify code reloads (specially in python notebooks).
    '''
    def print_debug_statement(self):
        print('This change must appear in the notebook')
        print('Testing reload functionality.')

'''
Function to compute the Rpr fairness penalty.
Parameters:
    y_pred_1 (jnp.ndarray): Predicted probabilities for class 1.
    y_pred_0 (jnp.ndarray): Predicted probabilities for class 0.
    S (jnp.ndarray): Sensitive attribute.
    eps (float): Small constant to avoid numerical issues.
Returns:
    float: Computed Rpr penalty. 
OBS:
    This function should only contain JAX operations to ensure compatibility with JAX automatic differentiation and
    JAX JIT compilation.
    Information about Rpr can be found in:
    'Fairness-Aware Classifier with Prejudice Remover Regularizer' - Kimishima et al.
'''
def compute_rpr(y_pred_1, y_pred_0, S, eps):
    p_hat_y1 = jnp.mean(y_pred_1)
    p_hat_y0 = jnp.mean(y_pred_0)

    mask0 = (S == 0)
    mask1 = (S == 1)
    count0 = jnp.sum(mask0)
    count1 = jnp.sum(mask1)
    div0 = jnp.maximum(count0, 1)
    div1 = jnp.maximum(count1, 1)

    p_hat_y0_given_s0 = jnp.sum(y_pred_0 * mask0) / div0
    p_hat_y0_given_s1 = jnp.sum(y_pred_0 * mask1) / div1
    p_hat_y1_given_s0 = jnp.sum(y_pred_1 * mask0) / div0
    p_hat_y1_given_s1 = jnp.sum(y_pred_1 * mask1) / div1

    ratio_y0_given_s0 = p_hat_y0_given_s0 / (p_hat_y0 + eps)
    ratio_y0_given_s1 = p_hat_y0_given_s1 / (p_hat_y0 + eps)
    ratio_y1_given_s0 = p_hat_y1_given_s0 / (p_hat_y1 + eps)
    ratio_y1_given_s1 = p_hat_y1_given_s1 / (p_hat_y1 + eps)

    log_ratio_y0 = jnp.where(mask0, jnp.log(ratio_y0_given_s0 + eps), jnp.log(ratio_y0_given_s1 + eps))
    log_ratio_y1 = jnp.where(mask0, jnp.log(ratio_y1_given_s0 + eps), jnp.log(ratio_y1_given_s1 + eps))

    rpr = jnp.mean(y_pred_1 * log_ratio_y1 + y_pred_0 * log_ratio_y0)
    return rpr