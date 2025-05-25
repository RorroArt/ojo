"""
KV Cache management.
"""
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from functools import partial
from einops import rearrange

@partial(jax.jit, donate_argnums=(0, 1))
def update_kv_cache(
    k_cache: jnp.ndarray, 
    v_cache: jnp.ndarray, 
    new_k: jnp.ndarray, 
    new_v: jnp.ndarray, 
    starting_pos: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Updates the KV cache with a new KV pair.

    Args:
        k_cache: The current KV cache for keys. [cache_len, NKVH, HD]
        v_cache: The current KV cache for values. [cache_len, NKVH, HD]
        new_k: The new key to add to the cache. [L, NKVH, HD]
        new_v: The new value to add to the cache. [L, NKVH, HD]
        starting_pos: The position to start updating the cache at. If None, starts from position 0.

    Returns:
        The updated KV cache.
    """
    if starting_pos is None:
        starting_pos = jnp.array(0, dtype=jnp.int32)
    
    k_updated = jax.lax.dynamic_update_slice_in_dim(k_cache, new_k, starting_pos, axis=0)
    v_updated = jax.lax.dynamic_update_slice_in_dim(v_cache, new_v, starting_pos, axis=0)
    
    return k_updated, v_updated


def extract_relevant_kv_cache(
    k_cache: jnp.ndarray, 
    v_cache: jnp.ndarray, 
    seq_len: int, 
    starting_pos: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extracts the relevant KV cache for the given sequence length, for attention computation.
    
    Args:
        k_cache: The KV cache for keys. [cache_len, NKVH, HD]
        v_cache: The KV cache for values. [cache_len, NKVH, HD]
        seq_len: The sequence length to extract the KV cache for.
        starting_pos: The position to start extracting the KV cache at. If None, starts from position 0.
        
    Returns:
        The relevant KV cache.

    """
    _, NKVH, HD = k_cache.shape

    if starting_pos is None:
        starting_pos = jnp.array(0, dtype=jnp.int32)
    
    relevant_len = starting_pos + seq_len

    k_relevant = jax.lax.dynamic_slice(k_cache, (0, 0, 0), (relevant_len, NKVH, HD))
    v_relevant = jax.lax.dynamic_slice(v_cache, (0, 0, 0), (relevant_len, NKVH, HD))

    return k_relevant, v_relevant




