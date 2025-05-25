# %%
import jax
import jax.numpy as jnp
from typing import NamedTuple

# %%

class AttentionParams(NamedTuple):
    W_q: jnp.ndarray  # Query weight matrix
    W_k: jnp.ndarray  # Key weight matrix  
    W_v: jnp.ndarray  # Value weight matrix
    W_o: jnp.ndarray  # Output projection

class MLPParams(NamedTuple):
    W1: jnp.ndarray   # First layer weights
    b1: jnp.ndarray   # First layer bias
    W2: jnp.ndarray   # Second layer weights  
    b2: jnp.ndarray   # Second layer bias

class ToyModelParams(NamedTuple):
    attention: AttentionParams
    mlp: MLPParams

def attention(params: AttentionParams, x: jnp.ndarray) -> jnp.ndarray:
    """
    Simple self-attention mechanism.
    
    Args:
        params: AttentionParams containing weight matrices
        x: Input tensor of shape (seq_len, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model)
    """
    # Compute queries, keys, values
    Q = x @ params.W_q  # (seq_len, d_model)
    K = x @ params.W_k  # (seq_len, d_model)
    V = x @ params.W_v  # (seq_len, d_model)
    
    # Attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.T / jnp.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    out = attn_weights @ V  # (seq_len, d_model)
    
    # Output projection
    return out @ params.W_o

def ffn(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
    """
    Simple feedforward network with ReLU activation.
    
    Args:
        params: MLPParams containing weights and biases
        x: Input tensor of shape (..., d_model)
    
    Returns:
        Output tensor of same shape as input
    """
    # First layer with ReLU
    hidden = jax.nn.relu(x @ params.W1 + params.b1)
    
    # Second layer
    output = hidden @ params.W2 + params.b2
    
    return output

def toy_model(params: ToyModelParams, x: jnp.ndarray) -> jnp.ndarray:
    """
    Simple transformer-like model: attention + FFN.
    
    Args:
        params: ToyModelParams containing attention and MLP parameters
        x: Input tensor of shape (seq_len, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model)
    """
    # Apply attention
    attn_out = attention(params.attention, x)
    
    # Residual connection
    x = x + attn_out
    
    # Apply FFN
    ffn_out = ffn(params.mlp, x)
    
    # Residual connection
    output = x + ffn_out
    
    return output

# %%
# Helper function to initialize parameters
def init_params(d_model: int = 64, d_ff: int = 256, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> ToyModelParams:
    """Initialize parameters for the toy model."""
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    
    # Initialize attention parameters
    attention_params = AttentionParams(
        W_q=jax.random.normal(k1, (d_model, d_model)) * 0.1,
        W_k=jax.random.normal(k2, (d_model, d_model)) * 0.1,
        W_v=jax.random.normal(k3, (d_model, d_model)) * 0.1,
        W_o=jax.random.normal(k4, (d_model, d_model)) * 0.1,
    )
    
    # Initialize MLP parameters
    mlp_params = MLPParams(
        W1=jax.random.normal(k5, (d_model, d_ff)) * 0.1,
        b1=jnp.zeros(d_ff),
        W2=jax.random.normal(k6, (d_ff, d_model)) * 0.1,
        b2=jnp.zeros(d_model),
    )
    
    return ToyModelParams(
        attention=attention_params,
        mlp=mlp_params
    )

# %%

# Initialize parameters and test data
params = init_params(d_model=64, d_ff=256)
x = jax.random.normal(jax.random.PRNGKey(0), (10, 64))  # (seq_len=10, d_model=64)
    
# Test attention
attn_out = attention(params.attention, x)
print(f"Attention output shape: {attn_out.shape}")

# Test FFN
ffn_out = ffn(params.mlp, x)
print(f"FFN output shape: {ffn_out.shape}")

# Test toy model
model_out = toy_model(params, x)
print(f"Toy model output shape: {model_out.shape}")

print("All functions work correctly!")
# %%

import ast
import inspect

source = inspect.getsource(toy_model)
tree = ast.parse(source)

print("Original AST:")
print(ast.dump(tree, indent=2))

# %%

jaxpr = jax.make_jaxpr(toy_model)(params, x)
print(jaxpr)

# %%

from typing import Callable

def patch_attention_ast(tree: ast.AST, f: Callable) -> ast.AST:
    """
    Patches AST to replace attention(params, x) calls with f(attention, params, x).
    
    Args:
        tree: The AST tree to patch
        f: The wrapper function (used for naming in the AST)
    
    Returns:
        New patched AST tree
    """
    import copy
    
    class AttentionPatcher(ast.NodeTransformer):
        def visit_Call(self, node):
            # First, recursively visit child nodes
            self.generic_visit(node)
            
            # Check if this is a call to 'attention'
            if (isinstance(node.func, ast.Name) and 
                node.func.id == 'attention'):
                
                # Create new call: f(attention, params, x)
                new_call = ast.Call(
                    func=ast.Name(id='f', ctx=ast.Load()),
                    args=[
                        ast.Name(id='attention', ctx=ast.Load()),  # Add attention as first arg
                        *node.args  # Spread the original arguments (params, x)
                    ],
                    keywords=node.keywords
                )
                return new_call
            
            return node
    
    # Create a deep copy of the tree to avoid modifying the original
    new_tree = copy.deepcopy(tree)
    patcher = AttentionPatcher()
    patched_tree = patcher.visit(new_tree)
    
    # Fix missing locations and return
    return ast.fix_missing_locations(patched_tree)

# %%
# Test the patching function
def dummy_wrapper(original_fn, *args, **kwargs):
    result = original_fn(*args, **kwargs)
    jax.debug.callback(lambda x: print(f"Attention output shape: {x.shape}"), result)
    return result

# Get the source and parse it
source = inspect.getsource(toy_model)
tree = ast.parse(source)

print("=== BEFORE PATCHING ===")
print(ast.unparse(tree))

# Apply the patch
patched_tree = patch_attention_ast(tree, dummy_wrapper)

print("\n=== AFTER PATCHING ===")
print(ast.unparse(patched_tree))

print("\n=== TEST SUCCESSFUL ===")
print("attention(params.attention, x) -> f(attention, params.attention, x)")
# %%

# run the patched function

# Compile the patched AST back to code
compiled_code = compile(patched_tree, filename="<patched>", mode="exec")

# Create a namespace with all the necessary functions and variables
namespace = {
    'AttentionParams': AttentionParams,
    'MLPParams': MLPParams,
    'ToyModelParams': ToyModelParams,
    'toy_model': toy_model,
    'attention': attention,
    'ffn': ffn,
    'f': dummy_wrapper,  # This is our wrapper function
    'jnp': jnp,
    'jax': jax
}

# Execute the patched code in our namespace
exec(compiled_code, namespace)

# Get the patched toy_model function from the namespace
patched_toy_model = namespace['toy_model']

print("=== RUNNING PATCHED FUNCTION ===")
# Run the patched function with our test parameters
result = patched_toy_model(params, x)
print(f"Final result shape: {result.shape}")
print("Patched function executed successfully!")
# %%

def recompile_ast(tree: ast.AST, function_name: str = None, extra_namespace: dict = None) -> callable:
    """
    Automatically recompiles an AST by figuring out the required namespace.
    
    Args:
        tree: The AST tree to compile
        function_name: Name of the function to extract (if None, returns the namespace)
        extra_namespace: Additional name mappings to include in the namespace
    
    Returns:
        The compiled function or the full namespace if function_name is None
    """
    import inspect
    
    # Get the current frame to access globals and locals
    frame = inspect.currentframe().f_back
    current_globals = frame.f_globals
    current_locals = frame.f_locals
    
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()
        
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):  # Only collect names that are being loaded/read
                self.names.add(node.id)
            self.generic_visit(node)
        
        def visit_Attribute(self, node):
            # For attributes like jnp.sqrt, collect the base name
            if isinstance(node.value, ast.Name):
                self.names.add(node.value.id)
            self.generic_visit(node)
    
    # Collect all names referenced in the AST
    collector = NameCollector()
    collector.visit(tree)
    
    # Build namespace by looking up names in current scope
    namespace = {}
    for name in collector.names:
        if name in current_locals:
            namespace[name] = current_locals[name]
        elif name in current_globals:
            namespace[name] = current_globals[name]
        else:
            # Skip names that aren't found (they might be defined within the function)
            pass
    
    # Also add common JAX/numpy names that might be needed
    common_names = ['jax', 'jnp', 'jax.nn', 'jax.random', 'jax.debug']
    for name in common_names:
        if name in current_globals:
            namespace[name] = current_globals[name]
    
    # Add extra namespace mappings if provided
    if extra_namespace:
        namespace.update(extra_namespace)
    
    print(f"Auto-detected namespace: {list(namespace.keys())}")
    
    # Compile and execute the AST
    compiled_code = compile(tree, filename="<recompiled>", mode="exec")
    exec(compiled_code, namespace)
    
    # Return the requested function or the full namespace
    if function_name:
        if function_name in namespace:
            return namespace[function_name]
        else:
            raise ValueError(f"Function '{function_name}' not found in compiled namespace")
    else:
        return namespace

# %%
# Test the recompile_ast function

print("=== TESTING RECOMPILE_AST ===")

# Get the patched AST from before
source = inspect.getsource(toy_model)
tree = ast.parse(source)
patched_tree = patch_attention_ast(tree, dummy_wrapper)

# Use the new recompile function with explicit mapping for 'f'
recompiled_toy_model = recompile_ast(patched_tree, 'toy_model', extra_namespace={'f': dummy_wrapper})

print("=== RUNNING RECOMPILED FUNCTION ===")
result = recompiled_toy_model(params, x)
print(f"Final result shape: {result.shape}")
print("Recompiled function executed successfully!")
# %%

def build_act_cache():
    cache = []
    
    def store_in_cache(value):
        """Helper function to store value in cache - called via pure_callback"""
        cache.append(value)
        return None
    
    def cache_wrapper(original_fn, *args, **kwargs):
        result = original_fn(*args, **kwargs)
        
        # Use pure_callback to store in cache without being traced
        jax.experimental.io_callback(
            store_in_cache, 
            None,  # return type (None since we're just storing)
            result,
        )
        
        return result
    
    # Return both the wrapper and a way to access the cache
    cache_wrapper.get_cache = lambda: cache
    return cache_wrapper

# %%

# Test the activation cache
print("=== TESTING ACTIVATION CACHE ===")

# Create a cache wrapper
cache_wrapper = build_act_cache()

# Test with our toy model
source = inspect.getsource(toy_model)
tree = ast.parse(source)
patched_tree = patch_attention_ast(tree, cache_wrapper)

# Recompile with the cache wrapper
cached_toy_model = jax.jit(recompile_ast(patched_tree, 'toy_model', extra_namespace={'f': cache_wrapper}))

print("Running cached toy model...")
result1 = cached_toy_model(params, x)
print(f"First run result shape: {result1.shape}")

# Check the cache
cache = cache_wrapper.get_cache()
print(f"Cache size: {len(cache)} entries")

# Run again to see if cache grows
result2 = cached_toy_model(params, x)
print(f"Second run result shape: {result2.shape}")

# Check cache again
cache = cache_wrapper.get_cache()
print(f"Cache size: {len(cache)} entries")

print("Cache test completed!")# %%
# %%

class MultiLayerToyModelParams(NamedTuple):
    q_weight: jnp.ndarray # (n_layers, d_model, d_model)
    k_weight: jnp.ndarray # (n_layers, d_model, d_model)
    v_weight: jnp.ndarray # (n_layers, d_model, d_model)
    o_weight: jnp.ndarray # (n_layers, d_model, d_model)
    mlp_weight: jnp.ndarray # (n_layers, d_model, d_ff)
    mlp_bias: jnp.ndarray # (n_layers, d_ff)
    mlp_output_weight: jnp.ndarray # (n_layers, d_ff, d_model)
    mlp_output_bias: jnp.ndarray # (n_layers, d_model)


def multilayer_attention(q_weight: jnp.ndarray, k_weight: jnp.ndarray, v_weight: jnp.ndarray, o_weight: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Multi-head self-attention for a single layer.
    
    Args:
        q_weight: Query weight matrix (d_model, d_model)
        k_weight: Key weight matrix (d_model, d_model)
        v_weight: Value weight matrix (d_model, d_model)
        o_weight: Output projection weight matrix (d_model, d_model)
        x: Input tensor (seq_len, d_model)
    
    Returns:
        Output tensor (seq_len, d_model)
    """
    # Compute queries, keys, values
    Q = x @ q_weight  # (seq_len, d_model)
    K = x @ k_weight  # (seq_len, d_model)
    V = x @ v_weight  # (seq_len, d_model)
    
    # Attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.T / jnp.sqrt(d_k)  # (seq_len, seq_len)
    
    # Apply softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply attention to values
    out = attn_weights @ V  # (seq_len, d_model)
    
    # Output projection
    return out @ o_weight

def multilayer_ffn(mlp_weight: jnp.ndarray, mlp_bias: jnp.ndarray, mlp_output_weight: jnp.ndarray, mlp_output_bias: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Feedforward network for a single layer.
    
    Args:
        mlp_weight: First layer weights (d_model, d_ff)
        mlp_bias: First layer bias (d_ff,)
        mlp_output_weight: Output layer weights (d_ff, d_model)
        mlp_output_bias: Output layer bias (d_model,)
        x: Input tensor (seq_len, d_model)
    
    Returns:
        Output tensor (seq_len, d_model)
    """
    # First layer with ReLU activation
    hidden = jax.nn.relu(x @ mlp_weight + mlp_bias)
    
    # Output layer
    output = hidden @ mlp_output_weight + mlp_output_bias
    
    return output

def multilayer_layer_fn(layer_params, x):
    """Process a single transformer layer"""
    # Unpack the accumulator and input
    x_input = x
    # Apply attention with residual connection
    attn_out = multilayer_attention(layer_params.q_weight, layer_params.k_weight, layer_params.v_weight, layer_params.o_weight, x_input)
    x_after_attn = x_input + attn_out
    
    # Apply FFN with residual connection
    ffn_out = multilayer_ffn(layer_params.mlp_weight, layer_params.mlp_bias, layer_params.mlp_output_weight, layer_params.mlp_output_bias, x_after_attn)
    x_output = x_after_attn + ffn_out
    
    # Return new accumulator and output
    return x_output  # (carry, output)

def multilayer_toy_model(params: MultiLayerToyModelParams, x: jnp.ndarray) -> jnp.ndarray:
    """
    Multi-layer transformer model using jax.lax.scan.
    
    Args:
        params: MultiLayerToyModelParams containing all layer parameters
        x: Input tensor (seq_len, d_model)
    
    Returns:
        Output tensor (seq_len, d_model)
    """
    
    # Scan over layers
    final_carry, _ = jax.lax.scan(lambda carry, lp: (multilayer_layer_fn(lp, carry), None), x, params)
    
    # Return just the output (not the accumulator)
    return final_carry

# Helper function to initialize multi-layer parameters
def init_multilayer_params(n_layers: int = 4, d_model: int = 64, d_ff: int = 256, 
                          key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> MultiLayerToyModelParams:
    """Initialize parameters for the multi-layer toy model."""
    keys = jax.random.split(key, 8)
    
    return MultiLayerToyModelParams(
        q_weight=jax.random.normal(keys[0], (n_layers, d_model, d_model)) * 0.1,
        k_weight=jax.random.normal(keys[1], (n_layers, d_model, d_model)) * 0.1,
        v_weight=jax.random.normal(keys[2], (n_layers, d_model, d_model)) * 0.1,
        o_weight=jax.random.normal(keys[3], (n_layers, d_model, d_model)) * 0.1,
        mlp_weight=jax.random.normal(keys[4], (n_layers, d_model, d_ff)) * 0.1,
        mlp_bias=jnp.zeros((n_layers, d_ff)),
        mlp_output_weight=jax.random.normal(keys[5], (n_layers, d_ff, d_model)) * 0.1,
        mlp_output_bias=jnp.zeros((n_layers, d_model))
    )

# Test the multi-layer model
print("=== TESTING MULTI-LAYER MODEL ===")
multilayer_params = init_multilayer_params(n_layers=4, d_model=64, d_ff=256)
test_x = jax.random.normal(jax.random.PRNGKey(123), (10, 64))  # (seq_len=10, d_model=64)

output = multilayer_toy_model(multilayer_params, test_x)
print(f"Multi-layer model output shape: {output.shape}")
print("Multi-layer model works correctly!")




    
    
# %%
import functools
def build_scan_cache(targets: list[int]):
    cache = []

    def store_in_cache(value):
        cache.append(value)
        return None
    
    def cache_wrapper(f, *args, **kwargs):
        result = f(*args, **kwargs)
        
        # Use pure_callback to store in cache without being traced
        jax.experimental.io_callback(
            store_in_cache,
            None,
            result
        )
        
        return result
    
    # Create a class to hold both the wrapper and cache accessor
    class CacheWrapper:
        def __init__(self, wrapper_fn):
            self.wrapper_fn = wrapper_fn
            self._cache = cache
        
        def __call__(self, f, *args, **kwargs):
            return self.wrapper_fn(f, *args, **kwargs)
        
        def get_cache(self):
            return self._cache
    
    return CacheWrapper(cache_wrapper)

def patch_scan_with_hook(tree, hook_fn):
    """Patches a scan operation to include an accumulator for hook registration."""
    # Convert tree to string
    code = ast.unparse(tree)
    
    # Define the patterns to replace
    old_scan = "jax.lax.scan(lambda carry, lp: (multilayer_layer_fn(lp, carry), None), x, params)"
    new_scan = """jax.lax.scan(
        lambda carry, lp: (
            (carry[0] + 1, hook_fn(multilayer_layer_fn, lp, carry[1])),
            None
        ),
        (jnp.array(0, dtype=jnp.int32), x),
        params
    )"""
    
    # Replace the scan operation
    patched_code = code.replace(old_scan, new_scan)
    
    # Parse back to AST
    return ast.parse(patched_code)

def test_multilayer_scan_hook():
    """
    Test the scan hook with our multilayer toy model.
    This demonstrates how to patch and use the scan hook in practice.
    """
    # Initialize model parameters
    n_layers = 4
    d_model = 64
    d_ff = 256
    params = init_multilayer_params(n_layers=n_layers, d_model=d_model, d_ff=d_ff)
    x = jax.random.normal(jax.random.PRNGKey(0), (10, d_model))  # (seq_len=10, d_model=64)
    
    # Create a cache wrapper for specific layers
    cache_wrapper = build_scan_cache([0, 2])  # Cache activations from layers 0 and 2
    
    # Get the source and parse it
    source = inspect.getsource(multilayer_toy_model)
    tree = ast.parse(source)
    
    print("\nOriginal code:")
    print(ast.unparse(tree))
    
    # Patch the scan operation
    patched_tree = patch_scan_with_hook(tree, cache_wrapper)
    
    print("\nPatched code:")
    print(ast.unparse(patched_tree))
    
    # Create namespace with all necessary functions
    namespace = {
        'jax': jax,
        'jnp': jnp,
        'multilayer_layer_fn': multilayer_layer_fn,
        'hook_fn': cache_wrapper,
        'MultiLayerToyModelParams': MultiLayerToyModelParams,
        'functools': functools
    }
    
    # Recompile the patched function
    patched_model = recompile_ast(patched_tree, 'multilayer_toy_model', 
                                extra_namespace=namespace)
    
    # Run the patched model
    print("\nRunning patched multilayer model...")
    result = patched_model(params, x)
    print(f"Model output shape: {result[0].shape}")
    
    # Get the cached activations
    cache = cache_wrapper.get_cache()
    print(f"Number of cached activations: {len(cache)}")
    for i, activation in enumerate(cache):
        print(f"Cached activation {i} shape: {activation.shape}")

# Run the test
print("\n=== TESTING MULTILAYER SCAN HOOK ===")
test_multilayer_scan_hook()
# %%

def build_q_projection_cache(targets: list[int]):
    """Builds a cache for Q projections from specific layers."""
    cache = {}

    def store_in_cache(layer_idx, value):
        if layer_idx in targets:
            cache[layer_idx] = value
        return None
    
    def cache_wrapper(f, layer_params, x):
        # Get the layer index from the carry
        layer_idx = jax.lax.index_in_dim(x[0], 0, keepdims=False)  # Get first element of array
        x_input = x[1]    # Second element is the actual input
        
        # Call the original function
        result = f(layer_params, x_input)
        
        # Store Q projection if we're in a target layer
        if layer_idx in targets:
            # Compute Q projection
            Q = x_input @ layer_params.q_weight
            
            # Store in cache via callback
            jax.experimental.io_callback(
                lambda q: store_in_cache(layer_idx, q),
                None,
                Q
            )
        
        return result
    
    # Create a class to hold both the wrapper and cache accessor
    class CacheWrapper:
        def __init__(self, wrapper_fn):
            self.wrapper_fn = wrapper_fn
            self._cache = cache
        
        def __call__(self, f, *args, **kwargs):
            return self.wrapper_fn(f, *args, **kwargs)
        
        def get_cache(self):
            return self._cache
    
    return CacheWrapper(cache_wrapper)

def patch_scan_with_hook(tree, hook_fn):
    """Patches a scan operation to include an accumulator for hook registration."""
    # Convert tree to string
    code = ast.unparse(tree)
    
    # Define the patterns to replace
    old_scan = "jax.lax.scan(lambda carry, lp: (multilayer_layer_fn(lp, carry), None), x, params)"
    new_scan = """jax.lax.scan(
        lambda carry, lp: (
            (carry[0] + 1, hook_fn(multilayer_layer_fn, lp, carry[1])),
            None
        ),
        (jnp.array(0, dtype=jnp.int32), x),
        params
    )"""
    
    # Replace the scan operation
    patched_code = code.replace(old_scan, new_scan)
    
    # Parse back to AST
    return ast.parse(patched_code)

def test_q_projection_hook():
    """
    Test the Q projection hook with our multilayer toy model.
    This demonstrates how to patch and use the hook in practice.
    """
    # Initialize model parameters
    n_layers = 4
    d_model = 64
    d_ff = 256
    params = init_multilayer_params(n_layers=n_layers, d_model=d_model, d_ff=d_ff)
    x = jax.random.normal(jax.random.PRNGKey(0), (10, d_model))  # (seq_len=10, d_model=64)
    
    # Create a cache wrapper for specific layers
    cache_wrapper = build_q_projection_cache([1, 2])  # Cache Q projections from layers 1 and 2
    
    # Get the source and parse it
    source = inspect.getsource(multilayer_toy_model)
    tree = ast.parse(source)
    
    # Patch the scan operation
    tree = patch_scan_with_hook(tree, cache_wrapper)
    
    # Create namespace with all necessary functions
    namespace = {
        'jax': jax,
        'jnp': jnp,
        'multilayer_layer_fn': multilayer_layer_fn,
        'hook_fn': cache_wrapper,
        'MultiLayerToyModelParams': MultiLayerToyModelParams,
        'functools': functools
    }
    
    # Recompile the patched function
    patched_model = recompile_ast(tree, 'multilayer_toy_model', 
                                extra_namespace=namespace)
    
    # Run the patched model
    print("\nRunning patched multilayer model...")
    result = patched_model(params, x)
    print(f"Model output shape: {result[0].shape}")
    
    # Get the cached activations
    cache = cache_wrapper.get_cache()
    print("\nCached Q projections:")
    for layer_idx, q_proj in cache.items():
        print(f"Layer {layer_idx} Q projection shape: {q_proj.shape}")

# Run the test
print("\n=== TESTING Q PROJECTION HOOK ===")
test_q_projection_hook()
# %%
