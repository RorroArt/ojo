"""
Attention mechanisms.
"""
import jax
import jax.numpy as jnp
from typing import Optional
from einops import rearrange

from functools import partial

from ojo.nn import update_kv_cache, extract_relevant_kv_cache
from ojo.nn import apply_rope

def gq_attention(
    x: jnp.ndarray,
    q_proj: jnp.ndarray,
    k_proj: jnp.ndarray,
    v_proj: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    o_proj: jnp.ndarray,
    cis_freq: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    starting_pos: Optional[jnp.ndarray] = None,
    scale: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Grouped-Query Attention implementation with KV cache.
    
    Args:
        x: Input tensor of shape [L, D]
        q_proj: Query projection matrix of shape [D, NH * HD]
        k_proj: Key projection matrix of shape [D, NKVH * HD]
        v_proj: Value projection matrix of shape [D, NKVH * HD]
        k_cache: Key cache of shape [cache_len, NKVH, HD]
        v_cache: Value cache of shape [cache_len, NKVH, HD]
        o_proj: Output projection matrix of shape [NH * HD, D]
        mask: Optional attention mask of shape [max_L, max_L]
        starting_pos: Position to start updating the KV cache at
        scale: Scaling factor for attention scores as jax array
    
    Returns:
        Tuple of (output, updated_k_cache, updated_v_cache)
        output: Output tensor of shape [L, D]
        updated_k_cache: Updated key cache
        updated_v_cache: Updated value cache
    """
    L, D = x.shape
    _, NKVH, HD = k_cache.shape
    NH = q_proj.shape[1] // HD
    G = NH // NKVH

    xq = jnp.einsum('ld,dk->lk', x, q_proj) # [L, NH * HD]
    xk = jnp.einsum('ld,dk->lk', x, k_proj) # [L, NKVH * HD]
    xv = jnp.einsum('ld,dk->lk', x, v_proj) # [L, NKVH * HD]

    xq = rearrange(xq, 'l (h k) -> l h k', h=NH, k=HD)
    xk = rearrange(xk, 'l (h k) -> l h k', h=NKVH, k=HD)
    xv = rearrange(xv, 'l (h k) -> l h k', h=NKVH, k=HD)

    xq = apply_rope(xq, cis_freq)
    xk = apply_rope(xk, cis_freq)

    new_k_cache, new_v_cache = update_kv_cache(k_cache, v_cache, xk, xv, starting_pos) # [cache_len, NKVH, HD]

    kr, vr = extract_relevant_kv_cache(new_k_cache, new_v_cache, L, starting_pos) # [L, NKVH, HD]

    if G > 1:
        kr = jnp.repeat(kr, G, axis=1) # [L, NH, HD]
        vr = jnp.repeat(vr, G, axis=1) # [L, NH, HD]

    xq = rearrange(xq, 'l h k -> h l k')
    kr = rearrange(kr, 'l h k -> h l k')
    vr = rearrange(vr, 'l h k -> h l k')

    if scale is None:
        scale = (1.0 / jnp.sqrt(HD)).astype(x.dtype)

    scores = jnp.einsum('hik,hjk->hij', xq, kr) * scale # [NH, L, L]

    if mask is not None:
        if starting_pos is None:
            scores = scores + mask[None, :L, :L]
        else:
            relevant_len = starting_pos + L
            scores = scores + mask[None, :L, :relevant_len]

    scores = jax.nn.softmax(scores, axis=-1) # [NH, L, L]

    out = jnp.einsum('hij,hjk->hik', scores, vr) # [NH, L, HD]      
    out = rearrange(out, 'h l k -> l (h k)') # [L, NH * HD]
    out = jnp.einsum('lk,kd->ld', out, o_proj) # [L, D]

    return out, new_k_cache, new_v_cache



