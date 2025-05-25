"""
KV Cache management.
"""
import jax
import jax.numpy as jnp

from functools import partial

partial(jax.jit, donate_argnums=(0, 1))
def update_kv_cache(k_cache, v_cache, new_k, new_v):
    """
    Updates the KV cache with a new KV pair.

    Args:
        k_cache: The current KV cache for keys. (max_seq_len, d_model)
        v_cache: The current KV cache for values. (max_seq_len, d_model)
        new_k: The new key to add to the cache. (d_model)
        new_v: The new value to add to the cache. (d_model)


    Returns:
        The updated KV cache.
    """
    #TODO
    return k_cache, v_cache