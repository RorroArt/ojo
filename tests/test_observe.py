import pytest
import jax
import jax.numpy as jnp
from ojo.interp import observe, observe_function


def test_basic_observe():
    """Test basic observe primitive functionality"""
    def simple_function(x):
        y = x + 1
        observed_y = observe(y, "intermediate_value")
        return observed_y * 2

    result = simple_function(5.0)
    assert result == 12.0


def test_observe_identity():
    """Test that observe returns the exact same value"""
    x = jnp.array([1.0, 2.0, 3.0])
    observed_x = observe(x, "test_array")
    
    assert jnp.array_equal(x, observed_x)
    assert x.shape == observed_x.shape
    assert x.dtype == observed_x.dtype


def test_observe_multiple_values():
    """Test observing multiple values in a function"""
    def multi_observe_function(x, y):
        sum_val = x + y
        observed_sum = observe(sum_val, "sum")
        
        product_val = x * y
        observed_product = observe(product_val, "product")
        
        return observed_sum, observed_product

    result = multi_observe_function(3.0, 4.0)
    assert result == (7.0, 12.0)


def test_observe_with_jit():
    """Test observe primitive with JIT compilation"""
    def jit_function(x):
        intermediate = x ** 2
        observed = observe(intermediate, "squared")
        return observed + 1

    jit_fn = jax.jit(jit_function)
    result = jit_fn(5.0)
    assert result == 26.0
    
    result2 = jit_fn(3.0)
    assert result2 == 10.0


def test_observe_with_gradient():
    """Test observe primitive with gradient computation"""
    def grad_function(x):
        y = x ** 3
        observed_y = observe(y, "cubic")
        return observed_y * 2

    grad_fn = jax.grad(grad_function)
    result = grad_fn(2.0)
    assert result == 24.0


def test_observe_with_vmap():
    """Test observe primitive with vmap"""
    def vmap_function(x):
        squared = x ** 2
        observed = observe(squared, "squared_batch")
        return observed + 10

    vmap_fn = jax.vmap(vmap_function)
    inputs = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = vmap_fn(inputs)
    expected = jnp.array([11.0, 14.0, 19.0, 26.0])
    assert jnp.allclose(result, expected)


def test_observe_vmap_multiple_args():
    """Test observe with vmap over multiple arguments"""
    def multi_arg_function(x, y):
        sum_val = x + y
        observed_sum = observe(sum_val, "sum_batch")
        product_val = x * y
        observed_product = observe(product_val, "product_batch")
        return observed_sum * observed_product

    vmap_fn = jax.vmap(multi_arg_function)
    x_vals = jnp.array([1.0, 2.0, 3.0])
    y_vals = jnp.array([4.0, 5.0, 6.0])
    result = vmap_fn(x_vals, y_vals)
    expected = jnp.array([20.0, 70.0, 162.0])
    assert jnp.allclose(result, expected)


def test_observe_high_dimensional():
    """Test observe with high-dimensional tensors"""
    def high_dim_function(x):
        mean_val = jnp.mean(x, axis=(1, 2))
        observed_mean = observe(mean_val, "mean_4d")
        
        max_val = jnp.max(observed_mean, axis=1)
        observed_max = observe(max_val, "max_2d")
        
        return observed_max

    batch, height, width, channels = 2, 3, 4, 5
    x = jnp.ones((batch, height, width, channels))
    
    result = high_dim_function(x)
    assert result.shape == (batch,)
    assert jnp.allclose(result, jnp.ones(batch))


def test_observe_complex_computation():
    """Test observe with complex mathematical operations"""
    def complex_function(x):
        matrix = jnp.outer(x, x)
        observed_matrix = observe(matrix, "outer_product")
        
        eigenvals = jnp.linalg.eigvals(observed_matrix)
        observed_eigenvals = observe(eigenvals, "eigenvalues")
        
        return jnp.sum(observed_eigenvals)

    x = jnp.array([1.0, 2.0, 3.0])
    result = complex_function(x)
    expected_trace = jnp.sum(x ** 2)
    assert jnp.allclose(result, expected_trace)


def test_observe_multiple_outputs():
    """Test observe with functions returning multiple outputs"""
    def multi_output_function(x, y):
        sum_val = x + y
        observed_sum = observe(sum_val, "sum")
        
        diff_val = x - y
        observed_diff = observe(diff_val, "difference")
        
        product_val = x * y
        observed_product = observe(product_val, "product")
        
        return observed_sum, observed_diff, observed_product

    result = multi_output_function(10.0, 3.0)
    assert result == (13.0, 7.0, 30.0)


def test_observe_with_control_flow():
    """Test observe with conditional operations"""
    def conditional_function(x):
        intermediate = x ** 2
        observed_intermediate = observe(intermediate, "squared")
        
        result = jax.lax.cond(
            observed_intermediate > 16.0,
            lambda: observe(observed_intermediate * 2, "large_branch"),
            lambda: observe(observed_intermediate + 5, "small_branch")
        )
        return result

    result_large = conditional_function(5.0)
    assert result_large == 50.0
    
    result_small = conditional_function(3.0)
    assert result_small == 14.0


def test_observe_with_scan():
    """Test observe with jax.lax.scan"""
    def scan_function(x):
        def scan_body(carry, x_elem):
            new_carry = carry + x_elem
            observed_carry = observe(new_carry, "scan_carry")
            return observed_carry, observed_carry
        
        init_carry = 0.0
        final_carry, outputs = jax.lax.scan(scan_body, init_carry, x)
        return final_carry, outputs

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    final_carry, outputs = scan_function(x)
    
    expected_outputs = jnp.array([1.0, 3.0, 6.0, 10.0])
    assert jnp.allclose(outputs, expected_outputs)
    assert final_carry == 10.0


def test_observe_nested_in_observe_function():
    """Test observe primitive inside observe_function decorator"""
    @observe_function
    def nested_function(x, y):
        sum_val = x + y
        observed_sum = observe(sum_val, "inner_sum")
        
        product_val = x * y
        observed_product = observe(product_val, "inner_product")
        
        result = observed_sum * observed_product
        observed_result = observe(result, "final_result")
        
        return observed_result

    result = nested_function(3.0, 4.0)
    assert result == 84.0


def test_observe_nested_complex():
    """Test complex nesting of observe and observe_function"""
    @observe_function
    def inner_function(x):
        squared = x ** 2
        observed_squared = observe(squared, "inner_squared")
        return observed_squared + 1

    @observe_function
    def outer_function(x, y):
        inner_result = inner_function(x)
        observed_inner = observe(inner_result, "inner_result")
        
        y_processed = y * 2
        observed_y = observe(y_processed, "processed_y")
        
        final_result = observed_inner + observed_y
        observed_final = observe(final_result, "final_nested")
        
        return observed_final

    result = outer_function(5.0, 3.0)
    assert result == 32.0


def test_observe_with_jit_and_vmap():
    """Test observe with combined JIT and vmap"""
    def combined_function(x, y):
        sum_val = x + y
        observed_sum = observe(sum_val, "jit_vmap_sum")
        
        product_val = x * y  
        observed_product = observe(product_val, "jit_vmap_product")
        
        return observed_sum ** 2 + observed_product

    vmap_fn = jax.vmap(combined_function)
    jit_vmap_fn = jax.jit(vmap_fn)
    
    x_vals = jnp.array([1.0, 2.0, 3.0])
    y_vals = jnp.array([4.0, 5.0, 6.0])
    
    result = jit_vmap_fn(x_vals, y_vals)
    
    expected = jnp.array([29.0, 59.0, 99.0])
    assert jnp.allclose(result, expected)


def test_observe_with_grad_and_vmap():
    """Test observe with combined gradient and vmap"""
    def grad_vmap_function(x):
        y = x ** 3
        observed_y = observe(y, "cubic_for_grad")
        return observed_y * 2

    grad_fn = jax.grad(grad_vmap_function)
    
    vmap_grad_fn = jax.vmap(grad_fn)
    
    x_vals = jnp.array([1.0, 2.0, 3.0])
    result = vmap_grad_fn(x_vals)
    
    expected = jnp.array([6.0, 24.0, 54.0])
    assert jnp.allclose(result, expected)


def test_observe_preserves_tree_structure():
    """Test that observe preserves complex tree structures"""
    def tree_function(data):
        processed_a = data['a'] * 2
        observed_a = observe(processed_a, "processed_a")
        
        processed_b = data['b']['nested'] + 10
        observed_b = observe(processed_b, "processed_b")
        
        return {
            'result_a': observed_a,
            'result_b': observed_b,
            'combined': observed_a + observed_b
        }

    input_data = {
        'a': jnp.array([1.0, 2.0, 3.0]),
        'b': {'nested': jnp.array([4.0, 5.0, 6.0])}
    }
    
    result = tree_function(input_data)
    
    expected_a = jnp.array([2.0, 4.0, 6.0])
    expected_b = jnp.array([14.0, 15.0, 16.0])
    expected_combined = jnp.array([16.0, 19.0, 22.0])
    
    assert jnp.allclose(result['result_a'], expected_a)
    assert jnp.allclose(result['result_b'], expected_b)
    assert jnp.allclose(result['combined'], expected_combined)


def test_observe_large_scale():
    """Test observe with large-scale computations"""
    def large_scale_function(x):
        matrix = jnp.reshape(x, (32, 32))
        observed_matrix = observe(matrix, "large_matrix")
        
        result_matrix = jnp.dot(observed_matrix, observed_matrix.T)
        observed_result = observe(result_matrix, "matrix_product")
        
        trace_val = jnp.trace(observed_result)
        observed_trace = observe(trace_val, "trace")
        
        return observed_trace

    x = jnp.ones(1024)
    result = large_scale_function(x)
    
    expected = 1024.0
    assert jnp.allclose(result, expected)


def test_observe_error_handling():
    """Test that observe handles edge cases properly"""
    scalar = jnp.array(5.0)
    observed_scalar = observe(scalar, "scalar")
    assert observed_scalar == 5.0
    
    empty = jnp.array([])
    observed_empty = observe(empty, "empty")
    assert observed_empty.shape == (0,)
    
    complex_val = jnp.array([1.0 + 2.0j, 3.0 + 4.0j])
    observed_complex = observe(complex_val, "complex")
    assert jnp.allclose(observed_complex, complex_val)