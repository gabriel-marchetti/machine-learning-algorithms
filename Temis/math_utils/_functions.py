import numpy as np
import jax.numpy as jnp

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
