import pytest
import jax
import jax.numpy as jnp
from ojo.interp import observe_function


def test_basic_execution():
    """Test basic execution of observe_function"""
    @observe_function
    def add_one(x):
        return x + 1

    result = add_one(5.0)
    assert result == 6.0


def test_jit_compilation():
    """Test JIT compilation with observe_function"""
    @observe_function
    def multiply_by_two(x):
        return x * 2

    jit_fn = jax.jit(multiply_by_two)
    result = jit_fn(3.0)
    assert result == 6.0


def test_gradient_computation():
    """Test gradient computation with observe_function"""
    @observe_function
    def quadratic(x):
        return x ** 2

    grad_fn = jax.grad(quadratic)
    result = grad_fn(3.0)
    assert result == 6.0


def test_function_boundaries():
    """Test function boundaries in JAXPRs"""
    @observe_function
    def simple_function(x):
        return x + 1

    jaxpr = jax.make_jaxpr(simple_function)(5.0)
    assert "observe_function" in str(jaxpr)


def test_multiple_outputs():
    """Test functions with multiple outputs"""
    @observe_function
    def multi_output(x):
        return x + 1, x * 2, x ** 2

    result = multi_output(3.0)
    assert result == (4.0, 6.0, 9.0)


def test_basic_vmap():
    """Test basic vmap functionality"""
    @observe_function
    def add_five(x):
        return x + 5

    vmap_fn = jax.vmap(add_five)
    inputs = jnp.array([1.0, 2.0, 3.0])
    result = vmap_fn(inputs)
    expected = jnp.array([6.0, 7.0, 8.0])
    assert jnp.allclose(result, expected)


def test_vmap_multiple_args():
    """Test vmap with multiple arguments"""
    @observe_function
    def add_two_numbers(x, y):
        return x + y

    vmap_fn = jax.vmap(add_two_numbers)
    x_vals = jnp.array([1.0, 2.0, 3.0])
    y_vals = jnp.array([10.0, 20.0, 30.0])
    result = vmap_fn(x_vals, y_vals)
    expected = jnp.array([11.0, 22.0, 33.0])
    assert jnp.allclose(result, expected)


def test_vmap_mixed_batch_axes():
    """Test vmap with mixed batch axes"""
    @observe_function
    def add_broadcast(x, y):
        return x + y

    vmap_fn = jax.vmap(add_broadcast, in_axes=(0, None))
    x_vals = jnp.array([1.0, 2.0, 3.0])
    y_scalar = 10.0
    result = vmap_fn(x_vals, y_scalar)
    expected = jnp.array([11.0, 12.0, 13.0])
    assert jnp.allclose(result, expected)


def test_higher_order_vmap():
    """Test higher-order vmap (vmap of vmap)"""
    @observe_function
    def matrix_add(x, y):
        return x + y

    batch_size = 2
    inner_size = 3
    x_vals = jnp.ones((batch_size, inner_size))
    y_vals = jnp.ones((batch_size, inner_size)) * 20
    
    double_vmap = jax.vmap(jax.vmap(matrix_add))
    result = double_vmap(x_vals, y_vals)
    expected_shape = (batch_size, inner_size)
    expected = jnp.array([[21.0, 21.0, 21.0],
                         [21.0, 21.0, 21.0]])
    
    assert result.shape == expected_shape
    assert jnp.allclose(result, expected)


def test_complex_neural_network():
    """Test complex neural network-like operations"""
    @observe_function
    def mini_transformer_layer(x, weights_q, weights_k, weights_v, weights_ff):
        """Simplified transformer layer with multiple outputs"""
        q = jnp.dot(x, weights_q)
        k = jnp.dot(x, weights_k)
        v = jnp.dot(x, weights_v)
        
        attention_scores = jnp.dot(q, k.T)
        attention_output = jnp.dot(attention_scores, v)
        
        ff_output = jnp.tanh(jnp.dot(attention_output, weights_ff))
        
        return ff_output, attention_output, attention_scores

    seq_len, d_model, d_ff = 4, 6, 8
    
    x = jnp.ones((seq_len, d_model))
    weights_q = jnp.ones((d_model, d_model)) * 0.1
    weights_k = jnp.ones((d_model, d_model)) * 0.1
    weights_v = jnp.ones((d_model, d_model)) * 0.1
    weights_ff = jnp.ones((d_model, d_ff)) * 0.1
    
    result = mini_transformer_layer(x, weights_q, weights_k, weights_v, weights_ff)
    
    assert result[0].shape == (seq_len, d_ff)
    assert result[1].shape == (seq_len, d_model)
    assert result[2].shape == (seq_len, seq_len)
    
    batch_size = 3
    batched_x = jnp.ones((batch_size, seq_len, d_model))
    
    vmap_layer = jax.vmap(mini_transformer_layer, in_axes=(0, None, None, None, None))
    batched_result = vmap_layer(batched_x, weights_q, weights_k, weights_v, weights_ff)
    
    assert batched_result[0].shape == (batch_size, seq_len, d_ff)
    assert batched_result[1].shape == (batch_size, seq_len, d_model)
    assert batched_result[2].shape == (batch_size, seq_len, seq_len)
    
    jit_layer = jax.jit(mini_transformer_layer)
    jit_result = jit_layer(x, weights_q, weights_k, weights_v, weights_ff)
    assert jit_result[0].shape == (seq_len, d_ff)


def test_very_high_dimensional():
    """Test very high dimensional tensors"""
    @observe_function
    def high_dim_operation(x):
        """Operations on very high dimensional tensors"""
        pooled = jnp.mean(x, axis=(2, 3, 4))
        channel_max = jnp.max(pooled, axis=1)
        reduced = jnp.sum(channel_max, axis=1)
        return pooled, channel_max, reduced

    batch, channels, depth, height, width, features = 2, 4, 3, 5, 5, 8
    x = jnp.ones((batch, channels, depth, height, width, features))
    
    result = high_dim_operation(x)
    
    assert result[0].shape == (batch, channels, features)
    assert result[1].shape == (batch, features)
    assert result[2].shape == (batch,)
    
    extra_batch = 3
    extra_batched_x = jnp.ones((extra_batch, batch, channels, depth, height, width, features))
    
    vmap_fn = jax.vmap(high_dim_operation)
    extra_batched_result = vmap_fn(extra_batched_x)
    
    assert extra_batched_result[0].shape == (extra_batch, batch, channels, features)
    assert extra_batched_result[1].shape == (extra_batch, batch, features)
    assert extra_batched_result[2].shape == (extra_batch, batch)
    
    @observe_function
    def ultra_high_dim_op(x):
        return jnp.mean(x, axis=(1, 2, 3, 4, 5, 6))
    
    dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8 = 2, 2, 2, 3, 3, 2, 4, 2
    x_8d = jnp.ones((dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8))
    
    result_8d = ultra_high_dim_op(x_8d)
    assert result_8d.shape == (dim1, dim8)
    
    batch_8d = 2
    batched_x_8d = jnp.ones((batch_8d, dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8))
    vmap_8d = jax.vmap(ultra_high_dim_op)
    batched_result_8d = vmap_8d(batched_x_8d)
    
    assert batched_result_8d.shape == (batch_8d, dim1, dim8)


def test_combined_transformations():
    """Test combined transformations stress test"""
    @observe_function
    def complex_function(x, y):
        return jnp.sin(x * y), jnp.cos(x + y), x * y

    jit_vmap = jax.jit(jax.vmap(complex_function))
    x_vals = jnp.array([1.0, 2.0, 3.0])
    y_vals = jnp.array([0.5, 1.0, 1.5])
    result1 = jit_vmap(x_vals, y_vals)
    assert all(r.shape == (3,) for r in result1)
    
    grad_fn = jax.grad(lambda x, y: complex_function(x, y)[0])
    vmap_grad = jax.vmap(grad_fn, in_axes=(0, 0))
    result2 = vmap_grad(x_vals, y_vals)
    assert result2.shape == (3,)
    
    jit_vmap_grad = jax.jit(jax.vmap(grad_fn, in_axes=(0, 0)))
    result3 = jit_vmap_grad(x_vals, y_vals)
    assert result3.shape == (3,)


def test_nested_observed_functions():
    """Test nested observed functions stress test"""
    @observe_function
    def level1(x):
        return x + 1

    @observe_function
    def level2(x):
        return level1(x) * 2

    @observe_function
    def level3(x):
        return level2(x) ** 2, level1(x) ** 3

    result = level3(2.0)
    assert abs(result[0] - 36.0) < 1e-6
    assert abs(result[1] - 27.0) < 1e-6
    
    vmap_nested = jax.vmap(level3)
    inputs = jnp.array([1.0, 2.0, 3.0])
    vmap_result = vmap_nested(inputs)
    assert all(r.shape == (3,) for r in vmap_result)
    
    jit_nested = jax.jit(level3)
    jit_result = jit_nested(2.0)
    assert abs(jit_result[0] - 36.0) < 1e-6
    assert abs(jit_result[1] - 27.0) < 1e-6
    
    jaxpr = jax.make_jaxpr(level3)(2.0)
    boundary_count = str(jaxpr).count("observe_function")
    assert boundary_count >= 3