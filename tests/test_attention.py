"""
Unit tests for grouped-query attention functionality.
"""
import jax
import jax.numpy as jnp
import pytest
from ojo.nn.attention import gq_attention


class TestGQAttention:
    
    def test_gq_attention_basic_functionality(self):
        """Test basic GQ attention functionality with simple inputs."""
        L, D = 4, 16
        NH, NKVH, HD = 8, 2, 8
        cache_len = 10
        
        # Input
        x = jnp.ones((L, D))
        
        # Projection matrices
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        # KV cache
        k_cache = jnp.zeros((cache_len, NKVH, HD))
        v_cache = jnp.zeros((cache_len, NKVH, HD))
        
        # Run attention
        output, updated_k_cache, updated_v_cache = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj
        )
        
        # Check output shape
        assert output.shape == (L, D)
        
        # Check that cache was updated
        assert updated_k_cache.shape == (cache_len, NKVH, HD)
        assert updated_v_cache.shape == (cache_len, NKVH, HD)
        
        # Check that new keys/values were added to cache
        assert not jnp.allclose(updated_k_cache[:L], 0)
        assert not jnp.allclose(updated_v_cache[:L], 0)
    
    def test_gq_attention_with_starting_pos(self):
        """Test GQ attention with specific starting position in cache."""
        L, D = 3, 12
        NH, NKVH, HD = 6, 3, 4
        cache_len = 8
        starting_pos = 2
        
        x = jnp.ones((L, D)) * 0.5
        
        q_proj = jnp.ones((D, NH * HD)) * 0.2
        k_proj = jnp.ones((D, NKVH * HD)) * 0.2
        v_proj = jnp.ones((D, NKVH * HD)) * 0.2
        o_proj = jnp.ones((NH * HD, D)) * 0.2
        
        # Initialize cache with some existing values
        k_cache = jnp.ones((cache_len, NKVH, HD)) * 0.1
        v_cache = jnp.ones((cache_len, NKVH, HD)) * 0.1
        
        output, updated_k_cache, updated_v_cache = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj,
            starting_pos=starting_pos
        )
        
        # Check that cache was updated at the correct position
        assert not jnp.allclose(updated_k_cache[starting_pos:starting_pos+L], 
                               k_cache[starting_pos:starting_pos+L])
        assert not jnp.allclose(updated_v_cache[starting_pos:starting_pos+L], 
                               v_cache[starting_pos:starting_pos+L])
        
        # Check that other positions remained unchanged
        if starting_pos > 0:
            assert jnp.allclose(updated_k_cache[:starting_pos], k_cache[:starting_pos])
            assert jnp.allclose(updated_v_cache[:starting_pos], v_cache[:starting_pos])
    
    def test_gq_attention_with_mask(self):
        """Test GQ attention with attention mask."""
        L, D = 4, 8
        NH, NKVH, HD = 4, 2, 4
        cache_len = 6
        seq_len = 5
        
        x = jnp.ones((L, D))
        
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        k_cache = jnp.ones((cache_len, NKVH, HD))
        v_cache = jnp.ones((cache_len, NKVH, HD))
        
        # Create causal mask
        mask = jnp.triu(jnp.full((L, seq_len), -jnp.inf), k=1)
        
        output, _, _ = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj,
            mask=mask, seq_len=seq_len
        )
        
        assert output.shape == (L, D)
        assert jnp.isfinite(output).all()
    
    def test_gq_attention_with_custom_scale(self):
        """Test GQ attention with custom scaling factor."""
        L, D = 2, 8
        NH, NKVH, HD = 4, 2, 4
        cache_len = 4
        
        x = jnp.ones((L, D))
        
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        k_cache = jnp.zeros((cache_len, NKVH, HD))
        v_cache = jnp.zeros((cache_len, NKVH, HD))
        
        custom_scale = jnp.array(0.25)
        
        output, _, _ = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj,
            scale=custom_scale
        )
        
        assert output.shape == (L, D)
        assert jnp.isfinite(output).all()
    
    def test_gq_attention_grouped_heads(self):
        """Test that GQ attention correctly handles grouped query heads."""
        L, D = 2, 16
        NH, NKVH, HD = 8, 4, 4  # 2 query heads per key-value head
        cache_len = 4
        
        x = jnp.ones((L, D))
        
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        k_cache = jnp.zeros((cache_len, NKVH, HD))
        v_cache = jnp.zeros((cache_len, NKVH, HD))
        
        output, updated_k_cache, updated_v_cache = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj
        )
        
        # Check that grouping ratio is correct
        G = NH // NKVH
        assert G == 2
        
        # Check output shape
        assert output.shape == (L, D)
        
        # Check cache shapes
        assert updated_k_cache.shape == (cache_len, NKVH, HD)
        assert updated_v_cache.shape == (cache_len, NKVH, HD)
    
    def test_gq_attention_with_seq_len(self):
        """Test GQ attention with explicit sequence length."""
        L, D = 3, 8
        NH, NKVH, HD = 4, 2, 4
        cache_len = 10
        seq_len = 6
        
        x = jnp.ones((L, D))
        
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        # Fill cache with some values
        k_cache = jnp.ones((cache_len, NKVH, HD)) * 0.2
        v_cache = jnp.ones((cache_len, NKVH, HD)) * 0.3
        
        output, _, _ = gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj,
            seq_len=seq_len
        )
        
        assert output.shape == (L, D)
        assert jnp.isfinite(output).all()
    
    def test_gq_attention_dtype_preservation(self):
        """Test that GQ attention preserves input data types."""
        L, D = 2, 8
        NH, NKVH, HD = 4, 2, 4
        cache_len = 4
        
        for dtype in [jnp.float32, jnp.float16]:
            x = jnp.ones((L, D), dtype=dtype)
            
            q_proj = jnp.ones((D, NH * HD), dtype=dtype) * 0.1
            k_proj = jnp.ones((D, NKVH * HD), dtype=dtype) * 0.1
            v_proj = jnp.ones((D, NKVH * HD), dtype=dtype) * 0.1
            o_proj = jnp.ones((NH * HD, D), dtype=dtype) * 0.1
            
            k_cache = jnp.zeros((cache_len, NKVH, HD), dtype=dtype)
            v_cache = jnp.zeros((cache_len, NKVH, HD), dtype=dtype)
            
            output, updated_k_cache, updated_v_cache = gq_attention(
                x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj
            )
            
            assert output.dtype == dtype
            assert updated_k_cache.dtype == dtype
            assert updated_v_cache.dtype == dtype
    
    def test_gq_attention_jit_compilation(self):
        """Test that GQ attention works correctly when JIT compiled."""
        L, D = 2, 8
        NH, NKVH, HD = 4, 2, 4
        cache_len = 4
        
        x = jnp.ones((L, D))
        
        q_proj = jnp.ones((D, NH * HD)) * 0.1
        k_proj = jnp.ones((D, NKVH * HD)) * 0.1
        v_proj = jnp.ones((D, NKVH * HD)) * 0.1
        o_proj = jnp.ones((NH * HD, D)) * 0.1
        
        k_cache = jnp.zeros((cache_len, NKVH, HD))
        v_cache = jnp.zeros((cache_len, NKVH, HD))
        
        # The function is already JIT compiled, but test it explicitly
        jit_gq_attention = jax.jit(gq_attention, donate_argnums=(4, 5))
        
        output, updated_k_cache, updated_v_cache = jit_gq_attention(
            x, q_proj, k_proj, v_proj, k_cache, v_cache, o_proj
        )
        
        assert output.shape == (L, D)
        assert updated_k_cache.shape == (cache_len, NKVH, HD)
        assert updated_v_cache.shape == (cache_len, NKVH, HD)
        assert jnp.isfinite(output).all()