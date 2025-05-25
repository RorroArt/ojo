"""
Attention mechanisms.
"""
import jax
import jax.numpy as jnp

from einops import rearrange

from functools import partial

from ojo.nn import update_kv_cache


def gq_attention(
    x: jnp.ndarray,
    q_proj: jnp.ndarray,
    k_proj: jnp.ndarray,
    v_proj: jnp.ndarray,
    k_cache: jnp.ndarray,
    v_cache: jnp.ndarray,
    o_proj: jnp.ndarray,
    mask: jnp.ndarray = None,
    starting_pos: jnp.ndarray = None,
    scale: jnp.ndarray = None,
    seq_len: jnp.ndarray = None,
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
        mask: Optional attention mask of shape [L, L]
        starting_pos: Position to start updating the KV cache at
        scale: Scaling factor for attention scores as jax array
        seq_len: Current sequence length in the KV cache
    
    Returns:
        Tuple of (output, updated_k_cache, updated_v_cache)
        output: Output tensor of shape [L, D]
        updated_k_cache: Updated key cache
        updated_v_cache: Updated value cache
    """
    L, D = x.shape
    
    # Calculate head dimensions
    if NH := q_proj.shape[-1] // (k_proj.shape[-1] // v_proj.shape[-1]):
        NKVH = v_proj.shape[-1] // (q_proj.shape[-1] // NH)
        HD = q_proj.shape[-1] // NH
    else:
        # Fallback calculation
        NKVH = k_proj.shape[-1] // v_proj.shape[-1] * v_proj.shape[-1] // k_proj.shape[-1]
        HD = k_proj.shape[-1] // NKVH 
        NH = q_proj.shape[-1] // HD
    
    # Project input to q, k, v
    q = jnp.einsum('ld,dh->lh', x, q_proj)  # [L, NH * HD]
    k = jnp.einsum('ld,dh->lh', x, k_proj)  # [L, NKVH * HD]
    v = jnp.einsum('ld,dh->lh', x, v_proj)  # [L, NKVH * HD]
    
    # Reshape for multi-head attention
    q = rearrange(q, 'l (nh hd) -> l nh hd', nh=NH, hd=HD)
    
    # Ensure key and value have right shape for kv cache
    if k_cache.shape[1:] != (NKVH, HD):
        # Shape mismatch - reshape to match cache
        nkvh, hd = k_cache.shape[1], k_cache.shape[2]
        nk = rearrange(k, 'l (nkvh hd) -> l nkvh hd', nkvh=nkvh, hd=hd)
        nv = rearrange(v, 'l (nkvh hd) -> l nkvh hd', nkvh=nkvh, hd=hd)
    else:
        # Normal case - use calculated dimensions
        nk = rearrange(k, 'l (nkvh hd) -> l nkvh hd', nkvh=NKVH, hd=HD)
        nv = rearrange(v, 'l (nkvh hd) -> l nkvh hd', nkvh=NKVH, hd=HD)
    
    # Get cache dimensions and update KV cache
    cache_len = k_cache.shape[0]
    
    # Update KV cache
    updated_k_cache, updated_v_cache = update_kv_cache(
        k_cache, v_cache, nk, nv, starting_pos)
    
    # Handle effective sequence length - avoid computing attention over zeros
    if seq_len is None:
        # Default to full cache if seq_len not provided
        effective_len = cache_len
    else:
        # Use provided seq_len but ensure it's within cache bounds
        effective_len = jnp.minimum(seq_len, cache_len)
    
    # Create attention mask to avoid computing attention over padded positions
    # This is an XLA-friendly alternative to slicing
    position_ids = jnp.arange(cache_len)
    attn_mask = jnp.where(
        position_ids < effective_len,
        jnp.zeros((1,), dtype=x.dtype),
        jnp.full((1,), -1e10, dtype=x.dtype)
    )
    
    # Handle grouped-query attention
    G = NH // updated_k_cache.shape[1]
    k_repeated = jnp.repeat(updated_k_cache, G, axis=1)
    v_repeated = jnp.repeat(updated_v_cache, G, axis=1)
    
    # Set default scale if not provided
    if scale is None:
        scale = jnp.asarray(1.0 / jnp.sqrt(HD), dtype=x.dtype)
    
    # Compute attention scores
    scores = jnp.einsum('qnh,knh->qkn', q, k_repeated) * scale
    
    # Apply padding mask to avoid attention to padding positions
    # Shape: [1, cache_len] -> [1, cache_len, 1] -> [L, cache_len, NH]
    broadcast_mask = attn_mask.reshape(1, -1, 1)
    scores = scores + broadcast_mask
    
    # Apply user-provided attention mask if any
    if mask is not None:
        # Ensure mask fits the cache length
        effective_mask = mask
        if mask.shape[1] != cache_len:
            # Create mask with correct size using padding
            # This is XLA-compatible and handles any mask size
            padding = ((0, 0), (0, max(0, cache_len - mask.shape[1])))
            effective_mask = jnp.pad(
                mask[:, :min(mask.shape[1], cache_len)],
                padding,
                constant_values=-1e10
            )
        scores = scores + effective_mask[:, :, None]
    
    # Apply softmax and compute weighted sum
    attn_weights = jax.nn.softmax(scores, axis=1)
    out = jnp.einsum('qkn,knh->qnh', attn_weights, v_repeated)
    out = rearrange(out, 'l nh hd -> l (nh hd)')
    out = jnp.einsum('lh,hd->ld', out, o_proj)
    
    return out, updated_k_cache, updated_v_cache


