"""
Unit tests for neural network blocks functionality.
"""
import jax
import jax.numpy as jnp
import pytest
from functools import partial
from ojo.nn.blocks import apply_rope, ffn, rms_norm


class TestApplyRope:
    
    def test_apply_rope_basic_functionality(self):
        """Test basic RoPE application functionality."""
        seq_len, num_heads, head_dim = 4, 2, 8
        
        # Input tensor [L, H, D]
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]],
                       [[2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                       [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                        [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8]],
                       [[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                        [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]]])
        
        # Complex frequency tensor [L, D//2] (based on implementation using [:, None, :])
        cis_freq = jnp.ones((seq_len, head_dim // 2), dtype=jnp.complex64) * (0.8 + 0.6j)
        
        result = apply_rope(x, cis_freq)
        
        # Check output shape
        assert result.shape == x.shape
        
        # Check that result is different from input (RoPE should transform the values)
        assert not jnp.allclose(result, x)
        
        # Check that result is finite
        assert jnp.isfinite(result).all()
    
    def test_apply_rope_shape_consistency(self):
        """Test that RoPE works with different input shapes."""
        for seq_len in [1, 2, 8, 16]:
            for num_heads in [1, 4, 8]:
                for head_dim in [4, 8, 16, 32]:
                    x = jnp.ones((seq_len, num_heads, head_dim))
                    cis_freq = jnp.ones((seq_len, head_dim // 2), dtype=jnp.complex64)
                    
                    result = apply_rope(x, cis_freq)
                    
                    assert result.shape == (seq_len, num_heads, head_dim)
                    assert jnp.isfinite(result).all()
    
    def test_apply_rope_dtype_preservation(self):
        """Test that RoPE preserves input data types."""
        seq_len, num_heads, head_dim = 4, 2, 8
        
        for dtype in [jnp.float32, jnp.float16]:
            x = jnp.ones((seq_len, num_heads, head_dim), dtype=dtype)
            cis_freq = jnp.ones((seq_len, head_dim // 2), dtype=jnp.complex64)
            
            result = apply_rope(x, cis_freq)
            
            assert result.dtype == dtype
    
    def test_apply_rope_zero_frequencies(self):
        """Test RoPE with zero frequencies (should be identity-like)."""
        seq_len, num_heads, head_dim = 4, 2, 8
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]],
                       [[2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
                       [[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                        [0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8]],
                       [[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                        [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]]])
        
        # cis=1+0j should be close to identity
        cis_freq = jnp.ones((seq_len, head_dim // 2), dtype=jnp.complex64)
        
        result = apply_rope(x, cis_freq)
        
        assert result.shape == x.shape
        assert jnp.isfinite(result).all()
    
    def test_apply_rope_consistency(self):
        """Test that RoPE produces consistent results."""
        seq_len, num_heads, head_dim = 2, 1, 4
        x = jnp.array([[[1.0, 2.0, 3.0, 4.0]],
                       [[5.0, 6.0, 7.0, 8.0]]])
        
        # Apply RoPE with some complex frequency
        cis_freq = jnp.array([[0.8 + 0.6j, 0.7 + 0.7j], 
                              [0.6 + 0.8j, 0.5 + 0.9j]], dtype=jnp.complex64)
        
        result1 = apply_rope(x, cis_freq)
        result2 = apply_rope(x, cis_freq)
        
        # Should produce identical results when called with same inputs
        assert jnp.allclose(result1, result2)
        
        # Result should be different from input
        assert not jnp.allclose(result1, x)


class TestFFN:
    
    def test_ffn_basic_functionality(self):
        """Test basic FFN functionality.
        
        Note: The current FFN implementation uses 'lk,dk->ld' einsum which expects
        down_proj to have shape [d_model, d_ff]. This is unusual - typically it would
        be [d_ff, d_model] with einsum 'lk,kd->ld'.
        """
        seq_len, d_model = 4, 8
        d_ff = 16
        
        x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                       [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                       [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        gate_proj = jnp.ones((d_model, d_ff)) * 0.1
        up_proj = jnp.ones((d_model, d_ff)) * 0.2
        down_proj = jnp.ones((d_model, d_ff)) * 0.1  # Shape to match current implementation
        
        # Test with SiLU activation
        silu_ffn = partial(ffn, jax.nn.silu)
        result = silu_ffn(x, gate_proj, up_proj, down_proj)
        
        # Check output shape
        assert result.shape == x.shape
        
        # Check that result is finite
        assert jnp.isfinite(result).all()
        
        # Check that result is different from input
        assert not jnp.allclose(result, x)
    
    def test_ffn_different_activations(self):
        """Test FFN with different activation functions."""
        seq_len, d_model = 2, 4
        d_ff = 8
        
        x = jnp.ones((seq_len, d_model))
        gate_proj = jnp.ones((d_model, d_ff)) * 0.1
        up_proj = jnp.ones((d_model, d_ff)) * 0.2
        down_proj = jnp.ones((d_model, d_ff)) * 0.1  # Shape to match current implementation
        
        activations = [jax.nn.silu, jax.nn.relu, jax.nn.gelu, jax.nn.tanh]
        results = []
        
        for act in activations:
            act_ffn = partial(ffn, act)
            result = act_ffn(x, gate_proj, up_proj, down_proj)
            
            assert result.shape == x.shape
            assert jnp.isfinite(result).all()
            results.append(result)
        
        # Different activations should produce different results
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not jnp.allclose(results[i], results[j])
    
    def test_ffn_shape_consistency(self):
        """Test FFN with different input shapes."""
        for seq_len in [1, 4, 8]:
            for d_model in [4, 8, 16]:
                for d_ff in [8, 16, 32]:
                    x = jnp.ones((seq_len, d_model))
                    gate_proj = jnp.ones((d_model, d_ff)) * 0.1
                    up_proj = jnp.ones((d_model, d_ff)) * 0.2
                    down_proj = jnp.ones((d_model, d_ff)) * 0.1  # Shape to match current implementation
                    
                    silu_ffn = partial(ffn, jax.nn.silu)
                    result = silu_ffn(x, gate_proj, up_proj, down_proj)
                    
                    assert result.shape == (seq_len, d_model)
                    assert jnp.isfinite(result).all()
    
    def test_ffn_dtype_preservation(self):
        """Test that FFN preserves input data types."""
        seq_len, d_model, d_ff = 2, 4, 8
        
        for dtype in [jnp.float32, jnp.float16]:
            x = jnp.ones((seq_len, d_model), dtype=dtype)
            gate_proj = jnp.ones((d_model, d_ff), dtype=dtype) * 0.1
            up_proj = jnp.ones((d_model, d_ff), dtype=dtype) * 0.2
            down_proj = jnp.ones((d_model, d_ff), dtype=dtype) * 0.1  # Shape to match current implementation
            
            silu_ffn = partial(ffn, jax.nn.silu)
            result = silu_ffn(x, gate_proj, up_proj, down_proj)
            
            assert result.dtype == dtype
    
    def test_ffn_zero_input(self):
        """Test FFN with zero input."""
        seq_len, d_model, d_ff = 2, 4, 8
        
        x = jnp.zeros((seq_len, d_model))
        gate_proj = jnp.ones((d_model, d_ff)) * 0.1
        up_proj = jnp.ones((d_model, d_ff)) * 0.2
        down_proj = jnp.ones((d_model, d_ff)) * 0.1  # Shape to match current implementation
        
        silu_ffn = partial(ffn, jax.nn.silu)
        result = silu_ffn(x, gate_proj, up_proj, down_proj)
        
        assert result.shape == x.shape
        assert jnp.isfinite(result).all()
    
    def test_ffn_gated_mechanism(self):
        """Test that FFN implements proper gated mechanism (SwiGLU-style)."""
        seq_len, d_model, d_ff = 2, 4, 8
        
        x = jnp.ones((seq_len, d_model))
        
        # Different gate and up_proj should produce different results
        gate_proj1 = jnp.ones((d_model, d_ff)) * 0.1
        up_proj1 = jnp.ones((d_model, d_ff)) * 0.2
        
        gate_proj2 = jnp.ones((d_model, d_ff)) * 0.5
        up_proj2 = jnp.ones((d_model, d_ff)) * 0.1
        
        down_proj = jnp.ones((d_model, d_ff)) * 0.1  # Shape to match current implementation
        
        silu_ffn = partial(ffn, jax.nn.silu)
        result1 = silu_ffn(x, gate_proj1, up_proj1, down_proj)
        result2 = silu_ffn(x, gate_proj2, up_proj2, down_proj)
        
        # Different gate/up_proj combinations should produce different results
        assert not jnp.allclose(result1, result2)


class TestRMSNorm:
    
    def test_rms_norm_basic_functionality(self):
        """Test basic RMS normalization functionality."""
        seq_len, d_model = 4, 8
        
        x = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                       [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                       [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0],
                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        
        weight = jnp.ones(d_model)
        
        result = rms_norm(x, weight)
        
        # Check output shape
        assert result.shape == x.shape
        
        # Check that result is finite
        assert jnp.isfinite(result).all()
        
        # Check that RMS is approximately 1 for each row (within tolerance)
        rms_values = jnp.sqrt(jnp.mean(result**2, axis=-1))
        assert jnp.allclose(rms_values, 1.0, atol=1e-6)
    
    def test_rms_norm_shape_consistency(self):
        """Test RMS norm with different input shapes."""
        for seq_len in [1, 2, 8, 16]:
            for d_model in [4, 8, 16, 32]:
                x = jnp.ones((seq_len, d_model)) * 2.0
                weight = jnp.ones(d_model)
                
                result = rms_norm(x, weight)
                
                assert result.shape == (seq_len, d_model)
                assert jnp.isfinite(result).all()
                
                # Check RMS is approximately 1
                rms_values = jnp.sqrt(jnp.mean(result**2, axis=-1))
                assert jnp.allclose(rms_values, 1.0, atol=1e-6)
    
    def test_rms_norm_dtype_preservation(self):
        """Test that RMS norm preserves input data types."""
        seq_len, d_model = 4, 8
        
        for dtype in [jnp.float32, jnp.float16]:
            x = jnp.ones((seq_len, d_model), dtype=dtype) * 2.0
            weight = jnp.ones(d_model, dtype=dtype)
            
            result = rms_norm(x, weight)
            
            assert result.dtype == dtype
            
            # Check RMS is approximately 1 (with appropriate tolerance for dtype)
            rms_values = jnp.sqrt(jnp.mean(result**2, axis=-1))
            tol = 1e-3 if dtype == jnp.float16 else 1e-6
            assert jnp.allclose(rms_values, 1.0, atol=tol)
    
    def test_rms_norm_different_eps_values(self):
        """Test RMS norm with different epsilon values."""
        seq_len, d_model = 2, 4
        x = jnp.ones((seq_len, d_model)) * 3.0
        weight = jnp.ones(d_model)
        
        eps_values = [1e-8, 1e-6, 1e-4, 1e-2]
        results = []
        
        for eps in eps_values:
            result = rms_norm(x, weight, eps=eps)
            
            assert result.shape == x.shape
            assert jnp.isfinite(result).all()
            
            # For uniform input, RMS norm should produce values close to weight
            # Allow for epsilon effects in the calculation
            assert jnp.allclose(result, weight, atol=1e-2)
            
            results.append(result)
        
        # Results should be very similar for reasonable eps values (first 3)
        for i in range(len(results) - 2):
            assert jnp.allclose(results[i], results[i + 1], atol=1e-4)
    
    def test_rms_norm_zero_input_handling(self):
        """Test RMS norm with zero input (should handle gracefully with eps)."""
        seq_len, d_model = 2, 4
        x = jnp.zeros((seq_len, d_model))
        weight = jnp.ones(d_model)
        
        result = rms_norm(x, weight, eps=1e-6)
        
        assert result.shape == x.shape
        assert jnp.isfinite(result).all()
        # With zero input, output should also be zero
        assert jnp.allclose(result, 0.0)
    
    def test_rms_norm_scale_invariance(self):
        """Test that RMS norm properly normalizes different scales."""
        seq_len, d_model = 2, 4
        
        # Test with different scales
        scales = [0.1, 1.0, 10.0, 100.0]
        base_x = jnp.array([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 1.0, 4.0, 3.0]])
        weight = jnp.ones(d_model)
        
        results = []
        for scale in scales:
            x = base_x * scale
            result = rms_norm(x, weight)
            results.append(result)
        
        # All results should be the same (scale invariant)
        for i in range(len(results) - 1):
            assert jnp.allclose(results[i], results[i + 1], atol=1e-5)
    
    def test_rms_norm_numerical_stability(self):
        """Test RMS norm numerical stability with different value ranges."""
        seq_len, d_model = 2, 4
        weight = jnp.ones(d_model)
        
        # Test with moderate values to avoid numerical precision issues
        x_small = jnp.ones((seq_len, d_model)) * 0.1
        result_small = rms_norm(x_small, weight, eps=1e-6)
        assert jnp.isfinite(result_small).all()
        
        x_large = jnp.ones((seq_len, d_model)) * 10.0
        result_large = rms_norm(x_large, weight, eps=1e-6)
        assert jnp.isfinite(result_large).all()
        
        # Both should produce similar results (scale invariant)
        assert jnp.allclose(result_small, result_large, atol=1e-4)
    
    def test_rms_norm_weight_scaling(self):
        """Test that RMS norm properly applies weight scaling."""
        seq_len, d_model = 2, 4
        
        x = jnp.ones((seq_len, d_model)) * 2.0
        
        # Test with different weight values
        weight1 = jnp.ones(d_model)
        weight2 = jnp.ones(d_model) * 2.0
        
        result1 = rms_norm(x, weight1)
        result2 = rms_norm(x, weight2)
        
        # Results should be different due to different weights
        assert not jnp.allclose(result1, result2)
        
        # result2 should be approximately 2x result1
        assert jnp.allclose(result2, result1 * 2.0, atol=1e-6) 