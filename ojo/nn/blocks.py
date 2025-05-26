"""
Transformer blocks. Everything that is not attention :)
"""
import jax
import jax.numpy as jnp

from typing import Callable

from functools import partial
from einops import rearrange


def apply_rope(
    x: jnp.ndarray,
    cis_freq: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply Rotary Position Embedding (RoPE) to input tensor.
    
    Args:
        x: Input tensor of shape [L, H, D]
        cis_freq: Complex frequencies of shape [L, D//2]
    
    Returns:
        Tensor with RoPE applied of shape [L, H, D]
    """
    dtype = x.dtype
    x = x.astype(jnp.float32) # fp32 for better precision
    x = rearrange(x, 'l h (d r) -> l h d r', r=2) # [L, H, D/2, 2]
    x_c = jax.lax.complex(x[..., 0], x[..., 1]) 
    x_c = x_c * cis_freq[:, None, :]
    x = jnp.stack([jnp.real(x_c), jnp.imag(x_c)], axis=-1) # [L, H, D, 2]
    x = rearrange(x, 'l h d r -> l h (d r)') # [L, H, D*2]
    return x.astype(dtype)



# The activation parameter (act) should be partially applied.
def ffn(
    act: Callable,
    x: jnp.ndarray,
    gate_proj: jnp.ndarray,
    up_proj: jnp.ndarray,
    down_proj: jnp.ndarray,
) -> jnp.ndarray:
    """
    Feed-forward network with gated mechanism (SwiGLU-style).
    
    Args:
        act: Activation function (partially applied)
        x: Input tensor of shape [L, D]
        gate_proj: Gate projection weights of shape [D, FF]
        up_proj: Up projection weights of shape [D, FF]
        down_proj: Down projection weights of shape [D, FF]
    
    Returns:
        Output tensor of shape [L, D]
    """
    gate = act(jnp.einsum('ld,dk->lk', x, gate_proj))
    up = jnp.einsum('ld,dk->lk', x, up_proj)
    down = jnp.einsum('lk,dk->ld', gate*up, down_proj)
    return down


def rms_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Root Mean Square normalization.
    
    Args:
        x: Input tensor of shape [L, D]
        weight: Scale weights of shape [D]
        eps: Numerical stability constant
    
    Returns:
        Normalized tensor of shape [L, D]
    """
    ms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(ms + eps)
    return x * weight
