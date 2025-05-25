"""
Attention mechanisms.
"""
import jax
import jax.numpy as jnp

from einops import rearrange

from functools import partial

from ojo.nn import update_kv_cache


# TODO: RoPE, Make sure the cache is correctly implemented
def gq_attention(
    x: jnp.ndarray,
    q_proj: jnp.ndarray,
    k_proj: jnp.ndarray,
    v_proj: jnp.ndarray,
    o_proj: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    mask: jnp.ndarray = None,
    scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Grouped-Query Attention implementation with KV cache.
    
    Args:
        x: Input tensor of shape [L, D]
        q_proj: Query projection matrix of shape [D, NH * HD]
        k_proj: Key projection matrix of shape [D, NKVH * HD]
        v_proj: Value projection matrix of shape [D, NKVH * HD]
        o_proj: Output projection matrix of shape [NH * HD, D]
        k_cache: Key cache of shape [cache_len, NKVH, HD]
        v_cache: Value cache of shape [cache_len, NKVH, HD]
        mask: Optional attention mask of shape [L, L]
        scale: Scaling factor for attention scores (default: 1.0)
    
    Returns:
        Tuple of (output, updated_k_cache, updated_v_cache)
        output: Output tensor of shape [L, D]
        updated_k_cache: Updated key cache
        updated_v_cache: Updated value cache
    """
    L, D = x.shape
    NKVH = k_proj.shape[-1] // v_proj.shape[-1] * v_proj.shape[-1] // k_proj.shape[-1]
    HD = k_proj.shape[-1] // NKVH 
    NH = q_proj.shape[-1] // HD
    q = jnp.einsum('ld,dh->lh', x, q_proj)  # [L, NH * HD]
    k = jnp.einsum('ld,dh->lh', x, k_proj)  # [L, NKVH * HD]
    v = jnp.einsum('ld,dh->lh', x, v_proj)  # [L, NKVH * HD]
    
    # Reshape to separate heads
    q = rearrange(q, 'l (nkvh hd) -> l nkvh hd', nkvh=NKVH, hd=HD)  # [L, NKVH, HD]
    nk = rearrange(k, 'l (nkvh hd) -> l nkvh hd', nkvh=NKVH, hd=HD)  # [L, NKVH, HD]
    nv = rearrange(v, 'l (nkvh hd) -> l nkvh hd', nkvh=NKVH, hd=HD)  # [L, NKVH, HD]
    

    k,v = update_kv_cache(k_cache, v_cache, nk, nv)  # [cache_len + L, NKVH, HD]
    
    G = NH // NKVH
    k_repeated = jnp.repeat(k, G, axis=1)  # [seq_len, NH, HD]
    v_repeated = jnp.repeat(v, G, axis=1)  # [seq_len, NH, HD]
    q = rearrange(q, 'l nkvh hd -> l (nkvh g) hd', g=G)  # [L, NH, HD]
    
    scores = jnp.einsum('qnh,knh->qkn', q, k_repeated) * scale  # [L, seq_len, NH]
    if mask is not None:
        scores = scores + mask[:, :, None]
    attn_weights = jax.nn.softmax(scores, axis=1)
    out = jnp.einsum('qkn,knh->qnh', attn_weights, v_repeated)  # [L, NH, HD]
    out = rearrange(out, 'l nh hd -> l (nh hd)')  # [L, NH * HD]
    out = jnp.einsum('lh,hd->ld', out, o_proj)  # [L, D]
    
    return out, k, v




