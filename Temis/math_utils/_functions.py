'''
Module containing various mathematical utility functions. 
It defines activations functions utilized across different machine learning algorithms.
'''
import numpy as np
import jax.numpy as jnp

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))