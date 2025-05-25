"""
Unit tests for KV cache functionality.
"""
import jax
import jax.numpy as jnp
import pytest
from ojo.nn.kv_cache import update_kv_cache, extract_relevant_kv_cache


class TestKVCache:
    
    def test_update_kv_cache_with_starting_pos(self):
        """Test KV cache update with explicit starting position."""
        cache_len, nkvh, hd = 10, 4, 64
        new_len = 3
        starting_pos = jnp.array(2, dtype=jnp.int32)
        
        # Initialize cache
        k_cache = jnp.zeros((cache_len, nkvh, hd))
        v_cache = jnp.zeros((cache_len, nkvh, hd))
        
        # Create new key-value pairs
        new_k = jnp.ones((new_len, nkvh, hd))
        new_v = jnp.ones((new_len, nkvh, hd)) * 2
        
        # Update cache
        updated_k, updated_v = update_kv_cache(k_cache, v_cache, new_k, new_v, starting_pos)
        
        # Check that the cache was updated at the correct positions
        assert updated_k.shape == (cache_len, nkvh, hd)
        assert updated_v.shape == (cache_len, nkvh, hd)
        
        # Check that new values were inserted at starting_pos (left-to-right)
        assert jnp.allclose(updated_k[starting_pos:starting_pos+new_len], new_k)
        assert jnp.allclose(updated_v[starting_pos:starting_pos+new_len], new_v)
        
        # Check that other positions remain zero
        assert jnp.allclose(updated_k[:starting_pos], 0)
        assert jnp.allclose(updated_v[:starting_pos], 0)
        assert jnp.allclose(updated_k[starting_pos+new_len:], 0)
        assert jnp.allclose(updated_v[starting_pos+new_len:], 0)
    
    def test_update_kv_cache_without_starting_pos(self):
        """Test KV cache update without starting position (starts from position 0)."""
        cache_len, nkvh, hd = 8, 2, 32
        new_len = 5
        
        # Initialize cache with some values
        k_cache = jnp.ones((cache_len, nkvh, hd)) * 3
        v_cache = jnp.ones((cache_len, nkvh, hd)) * 4
        
        # Save copies of original cache for assertions (since arrays will be donated)
        k_cache_orig = k_cache.copy()
        v_cache_orig = v_cache.copy()
        
        # Create new key-value pairs
        new_k = jnp.ones((new_len, nkvh, hd)) * 7
        new_v = jnp.ones((new_len, nkvh, hd)) * 8
        
        # Update cache without starting_pos (defaults to 0)
        updated_k, updated_v = update_kv_cache(k_cache, v_cache, new_k, new_v, None)
        
        # Check shapes
        assert updated_k.shape == (cache_len, nkvh, hd)
        assert updated_v.shape == (cache_len, nkvh, hd)
        
        # Check that new values were inserted at the beginning (left-to-right from position 0)
        assert jnp.allclose(updated_k[:new_len], new_k)
        assert jnp.allclose(updated_v[:new_len], new_v)
        
        # Check that remaining values are from original cache
        assert jnp.allclose(updated_k[new_len:], k_cache_orig[new_len:])
        assert jnp.allclose(updated_v[new_len:], v_cache_orig[new_len:])
    
    def test_update_kv_cache_full_replacement(self):
        """Test KV cache update when new data fills entire cache."""
        cache_len, nkvh, hd = 6, 3, 16
        
        # Initialize cache
        k_cache = jnp.zeros((cache_len, nkvh, hd))
        v_cache = jnp.zeros((cache_len, nkvh, hd))
        
        # Create new key-value pairs that fill the entire cache
        new_k = jnp.ones((cache_len, nkvh, hd)) * 5
        new_v = jnp.ones((cache_len, nkvh, hd)) * 6
        
        # Update cache starting at position 0
        updated_k, updated_v = update_kv_cache(k_cache, v_cache, new_k, new_v, 0)
        
        # Check that entire cache was replaced
        assert jnp.allclose(updated_k, new_k)
        assert jnp.allclose(updated_v, new_v)
    
    def test_update_kv_cache_preserves_dtype(self):
        """Test that KV cache update preserves data types."""
        cache_len, nkvh, hd = 4, 2, 8
        new_len = 2
        
        # Test with different dtypes
        for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
            k_cache = jnp.zeros((cache_len, nkvh, hd), dtype=dtype)
            v_cache = jnp.zeros((cache_len, nkvh, hd), dtype=dtype)
            
            new_k = jnp.ones((new_len, nkvh, hd), dtype=dtype)
            new_v = jnp.ones((new_len, nkvh, hd), dtype=dtype)
            
            updated_k, updated_v = update_kv_cache(k_cache, v_cache, new_k, new_v, 1)
            
            assert updated_k.dtype == dtype
            assert updated_v.dtype == dtype
    
    def test_update_kv_cache_left_to_right_behavior(self):
        """Test that the function correctly implements left-to-right updating."""
        cache_len, nkvh, hd = 8, 2, 4
        
        # Initialize cache with a pattern to make updates visible (ensure float32 dtype)
        k_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd)
        v_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd) * 10
        
        # Save copies of original cache for assertions (since arrays will be donated)
        k_cache_orig = k_cache.copy()
        v_cache_orig = v_cache.copy()
        
        # Create new values with a different pattern
        new_len = 3
        starting_pos = jnp.array(2, dtype=jnp.int32)
        new_k = jnp.ones((new_len, nkvh, hd)) * 999
        new_v = jnp.ones((new_len, nkvh, hd)) * 888
        
        # Update cache
        updated_k, updated_v = update_kv_cache(k_cache, v_cache, new_k, new_v, starting_pos)
        
        # Verify left-to-right behavior: new values should appear at positions 2, 3, 4
        assert jnp.allclose(updated_k[starting_pos:starting_pos+new_len], new_k)
        assert jnp.allclose(updated_v[starting_pos:starting_pos+new_len], new_v)
        
        # Verify that positions before starting_pos are unchanged
        assert jnp.allclose(updated_k[:starting_pos], k_cache_orig[:starting_pos])
        assert jnp.allclose(updated_v[:starting_pos], v_cache_orig[:starting_pos])
        
        # Verify that positions after the update are unchanged
        assert jnp.allclose(updated_k[starting_pos+new_len:], k_cache_orig[starting_pos+new_len:])
        assert jnp.allclose(updated_v[starting_pos+new_len:], v_cache_orig[starting_pos+new_len:])
    
    def test_update_kv_cache_with_jit(self):
        """Test that the function works correctly when JIT compiled."""
        cache_len, nkvh, hd = 5, 2, 4
        new_len = 2
        starting_pos = jnp.array(1, dtype=jnp.int32)
        
        k_cache = jnp.zeros((cache_len, nkvh, hd))
        v_cache = jnp.zeros((cache_len, nkvh, hd))
        
        new_k = jnp.ones((new_len, nkvh, hd)) * 10
        new_v = jnp.ones((new_len, nkvh, hd)) * 20
        
        # The function is already JIT compiled, but let's test it explicitly
        jit_update = jax.jit(update_kv_cache, donate_argnums=(0, 1))
        updated_k, updated_v = jit_update(k_cache, v_cache, new_k, new_v, starting_pos)
        
        # Verify results
        assert jnp.allclose(updated_k[starting_pos:starting_pos+new_len], new_k)
        assert jnp.allclose(updated_v[starting_pos:starting_pos+new_len], new_v)
    
    def test_extract_relevant_kv_cache_basic(self):
        """Test basic functionality of extract_relevant_kv_cache."""
        cache_len, nkvh, hd = 10, 3, 8
        seq_len = 5
        starting_pos = jnp.array(2, dtype=jnp.int32)
        
        # Create cache with identifiable pattern
        k_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd)
        v_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd) * 10
        
        # Extract relevant cache
        k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, seq_len, starting_pos)
        
        # Check shapes - should be (starting_pos + seq_len, nkvh, hd)
        expected_len = starting_pos + seq_len
        assert k_relevant.shape == (expected_len, nkvh, hd)
        assert v_relevant.shape == (expected_len, nkvh, hd)
        
        # Check that extracted cache matches original cache up to relevant_len
        assert jnp.allclose(k_relevant, k_cache[:expected_len])
        assert jnp.allclose(v_relevant, v_cache[:expected_len])
    
    def test_extract_relevant_kv_cache_without_starting_pos(self):
        """Test extract_relevant_kv_cache without starting position (defaults to 0)."""
        cache_len, nkvh, hd = 8, 2, 4
        seq_len = 6
        
        # Create cache with pattern
        k_cache = jnp.ones((cache_len, nkvh, hd)) * 3
        v_cache = jnp.ones((cache_len, nkvh, hd)) * 7
        
        # Extract relevant cache without starting_pos
        k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, seq_len, None)
        
        # Check shapes - should be (0 + seq_len, nkvh, hd) = (seq_len, nkvh, hd)
        assert k_relevant.shape == (seq_len, nkvh, hd)
        assert v_relevant.shape == (seq_len, nkvh, hd)
        
        # Check that extracted cache matches original cache up to seq_len
        assert jnp.allclose(k_relevant, k_cache[:seq_len])
        assert jnp.allclose(v_relevant, v_cache[:seq_len])
    
    def test_extract_relevant_kv_cache_full_extraction(self):
        """Test extracting the entire cache."""
        cache_len, nkvh, hd = 5, 2, 3
        seq_len = 3
        starting_pos = jnp.array(2, dtype=jnp.int32)
        
        # Create cache
        k_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd)
        v_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd) * 2
        
        # Extract relevant cache (starting_pos + seq_len = 2 + 3 = 5, which is the full cache)
        k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, seq_len, starting_pos)
        
        # Should extract the entire cache
        assert k_relevant.shape == (cache_len, nkvh, hd)
        assert v_relevant.shape == (cache_len, nkvh, hd)
        assert jnp.allclose(k_relevant, k_cache)
        assert jnp.allclose(v_relevant, v_cache)
    
    def test_extract_relevant_kv_cache_preserves_dtype(self):
        """Test that extract_relevant_kv_cache preserves data types."""
        cache_len, nkvh, hd = 6, 2, 4
        seq_len = 3
        starting_pos = jnp.array(1, dtype=jnp.int32)
        
        # Test with different dtypes
        for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
            k_cache = jnp.ones((cache_len, nkvh, hd), dtype=dtype)
            v_cache = jnp.ones((cache_len, nkvh, hd), dtype=dtype)
            
            k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, seq_len, starting_pos)
            
            assert k_relevant.dtype == dtype
            assert v_relevant.dtype == dtype
    
    def test_extract_relevant_kv_cache_edge_cases(self):
        """Test edge cases for extract_relevant_kv_cache."""
        cache_len, nkvh, hd = 8, 2, 4
        
        # Test with seq_len = 1
        k_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd)
        v_cache = jnp.arange(cache_len * nkvh * hd, dtype=jnp.float32).reshape(cache_len, nkvh, hd) * 5
        
        k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, 1, jnp.array(3, dtype=jnp.int32))
        
        # Should extract up to position 3 + 1 = 4
        assert k_relevant.shape == (4, nkvh, hd)
        assert v_relevant.shape == (4, nkvh, hd)
        assert jnp.allclose(k_relevant, k_cache[:4])
        assert jnp.allclose(v_relevant, v_cache[:4])
        
        # Test with starting_pos = 0
        k_relevant, v_relevant = extract_relevant_kv_cache(k_cache, v_cache, 3, jnp.array(0, dtype=jnp.int32))
        
        # Should extract up to position 0 + 3 = 3
        assert k_relevant.shape == (3, nkvh, hd)
        assert v_relevant.shape == (3, nkvh, hd)
        assert jnp.allclose(k_relevant, k_cache[:3])
        assert jnp.allclose(v_relevant, v_cache[:3])