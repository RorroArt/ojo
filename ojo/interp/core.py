"""
Inspired by: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/harvest.py
"""


import jax
import jax.numpy as jnp
import jax.extend as jex
import jax._src.effects as effects
import jax._src.core as jc
from jax.interpreters import ad, batching, mlir
from jax._src.interpreters import partial_eval as pe
import jax.tree_util as tree_util
from jax._src import util, api_util, ad_util
from jax._src import source_info_util
from jax._src import linear_util as lu
from typing import List, Any, Dict, Callable

# ====== Observe value ======
# Primitive to add observations (tag) intermediate values within a jax function

obs_p = jex.core.Primitive("observe")
obs_p.multiple_results = True

# Handle effects 

class ObsEffect(effects.Effect):
    __repr__ = lambda _: "Observe" 

obs_effect = ObsEffect()

effects.remat_allowed_effects.add_type(ObsEffect)
effects.control_flow_allowed_effects.add_type(ObsEffect)
effects.lowerable_effects.add_type(ObsEffect)

# Abstract evaluation

@obs_p.def_impl
def _obs_impl(*args, **_):
    return args

@obs_p.def_effectful_abstract_eval
def _obs_abstract_eval(*avals, **_):
    return avals, {obs_effect}

# Automatic Differentiation

def _obs_jvp(primals, tangents, **kwargs):
    out_primals = obs_p.bind(*primals, **kwargs)
    return out_primals, tangents

ad.primitive_jvps[obs_p] = _obs_jvp

def _obs_transpose(cotangents, *args, **kwargs):
    del args, kwargs
    return cotangents

ad.primitive_transposes[obs_p] = _obs_transpose


# Batching

def _obs_batch_rule(batched_args, batch_dims, **params):
    outs = obs_p.bind(*batched_args, **params)
    return outs, batch_dims

batching.primitive_batchers[obs_p] = _obs_batch_rule

# MLIR lowering rule

mlir.register_lowering(obs_p, lambda ctx, *args, **kw : args)


def observe(value, name):
    with jc.take_current_trace() as trace:
        flat_args, in_tree = tree_util.tree_flatten(value)
        out_flat = obs_p.bind_with_trace(
            trace,
            flat_args,
            dict(name=name)
        )
        return tree_util.tree_unflatten(in_tree, out_flat)


# ====== Observe Function ======
# Primitive to add an observation (tag) to a function. It adds function limits within a jaxpr

obs_func_p = jex.core.Primitive("observe_function")
obs_func_p.multiple_results = True

class ObsEffect(effects.Effect):
    __repr__ = lambda _: "ObserveFunction"

obs_effect = ObsEffect()
effects.remat_allowed_effects.add_type(ObsEffect)
effects.control_flow_allowed_effects.add_type(ObsEffect)
effects.lowerable_effects.add_type(ObsEffect)

# Implementation and abstract evaluation

@obs_func_p.def_impl
def obs_func_impl(*args, jaxpr, **params):
    """Execute the jaxpr normally during interpretation"""
    return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)

@obs_func_p.def_effectful_abstract_eval
def obs_func_abstract_eval(*avals, jaxpr, **params):
    """Return the jaxpr's output types"""
    return jaxpr.out_avals, {obs_effect}

# MLIR lowering

# This is the approach that has worked. I'm not sure if it adds overhead. Thus, it shall be carefully profiled.
def obs_func_mlir_lowering(ctx, *args, jaxpr, **params):
    def jaxpr_func(*args):
        return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
    
    return mlir.lower_fun(jaxpr_func, multiple_results=True)(ctx, *args)

mlir.register_lowering(obs_func_p, obs_func_mlir_lowering)

# Autodiff

def obs_func_jvp(primals, tangents, *, jaxpr, **params):
    
    primal_out = obs_func_p.bind(*primals, jaxpr=jaxpr, **params)
    
    nz_tangents = []
    for t in tangents:
        if type(t) is not ad_util.Zero:
            nz_tangents.append(t)
    
    if not nz_tangents:
        if isinstance(primal_out, tuple):
            tangent_out = tuple(ad_util.Zero.from_primal_value(p) for p in primal_out)
        else:
            tangent_out = ad_util.Zero.from_primal_value(primal_out)
    else:
        def jaxpr_fun(*args):
            return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        
        actual_tangents = []
        for i, (p, t) in enumerate(zip(primals, tangents)):
            if type(t) is ad_util.Zero:
                actual_tangents.append(jnp.zeros_like(p))
            else:
                actual_tangents.append(t)
        
        _, tangent_out_raw = jax.jvp(jaxpr_fun, primals, tuple(actual_tangents))
        tangent_out = tangent_out_raw
    return primal_out, tangent_out


def obs_func_transpose(cts_out, *primals, jaxpr, **params):
    
    def jaxpr_fun(*args):
        return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
    
    _, vjp_fun = jax.vjp(jaxpr_fun, *primals)
    
    if isinstance(cts_out, tuple):
        if len(cts_out) == 1:
            result = vjp_fun(cts_out[0])
        else:
            result = vjp_fun(cts_out)
    else:
        result = vjp_fun(cts_out)
    
    if isinstance(result, tuple) and len(result) == 1:
        return result[0]
    else:
        return result

ad.primitive_jvps[obs_func_p] = obs_func_jvp
ad.primitive_transposes[obs_func_p] = obs_func_transpose


# Batching (vmap)

def obs_func_batch_rule(batched_args, batch_dims, *, jaxpr, **params):
    """Robust batch rule for vmap with multiple_results=True"""
    if all(bd is batching.not_mapped for bd in batch_dims):
        result = jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *batched_args)
        if isinstance(result, tuple):
            return result, tuple(batching.not_mapped for _ in result)
        else:
            return result, batching.not_mapped
    
    mapped_batch_dims = [bd for bd in batch_dims if bd is not batching.not_mapped]
    
    if not mapped_batch_dims:
        result = jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *batched_args)
        if isinstance(result, tuple):
            return result, tuple(batching.not_mapped for _ in result)
        else:
            return result, batching.not_mapped
    
    output_batch_dim = mapped_batch_dims[0]
    
    if not all(bd == output_batch_dim or bd is batching.not_mapped for bd in batch_dims):
        moved_args = []
        for arg, bd in zip(batched_args, batch_dims):
            if bd is not batching.not_mapped and bd != 0:
                moved_arg = jnp.moveaxis(arg, bd, 0)
                moved_args.append(moved_arg)
            else:
                moved_args.append(arg)
        
        def unbatched_fn(*args):
            return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        
        moved_batch_dims = tuple(0 if bd is not batching.not_mapped else batching.not_mapped 
                                for bd in batch_dims)
        vmapped_fn = jax.vmap(unbatched_fn, in_axes=moved_batch_dims)
        result = vmapped_fn(*moved_args)
        output_batch_dim = 0  # Result will have batch dim at 0
    else:
        def unbatched_fn(*args):
            return jc.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *args)
        
        vmapped_fn = jax.vmap(unbatched_fn, in_axes=batch_dims)
        result = vmapped_fn(*batched_args)
    
    if isinstance(result, (tuple, list)):
        result_batch_dims = tuple(output_batch_dim for _ in result)
        return result, result_batch_dims
    else:
        return result, output_batch_dim

batching.primitive_batchers[obs_func_p] = obs_func_batch_rule

def observe_function(fun: Callable, name: str = "observed"):
    @util.wraps(fun)
    def wrapped(*args, **kwargs):
        def traced_fun(*args, **kwargs):
            result = fun(*args, **kwargs)
            if isinstance(result, tuple):
                return result
            else:
                return (result,)
        
        debug = api_util.debug_info("observe_function", traced_fun, args, kwargs)
        wrapped_fun = lu.wrap_init(traced_fun, debug_info=debug)
        
        in_avals = [jc.get_aval(arg) for arg in args]
        if kwargs:
            kwarg_vals = list(kwargs.values())
            in_avals.extend([jc.get_aval(val) for val in kwarg_vals])
            def flat_traced_fun(*all_args):
                n_args = len(args)
                pos_args = all_args[:n_args]
                kwarg_vals = all_args[n_args:]
                kwarg_dict = dict(zip(kwargs.keys(), kwarg_vals))
                return traced_fun(*pos_args, **kwarg_dict)
            wrapped_fun = lu.wrap_init(flat_traced_fun, debug_info=debug)
            flat_args = list(args) + list(kwargs.values())
        else:
            flat_args = list(args)
        
        jaxpr, out_avals, consts, attrs_tracked = pe.trace_to_jaxpr_dynamic(wrapped_fun, in_avals)
        
        if any(isinstance(c, jc.Tracer) for c in consts):
            raise jc.UnexpectedTracerError(
                "Found a JAX Tracer as a constant in observe_function.")
        
        closed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
        
        out_flat = obs_func_p.bind(*flat_args, jaxpr=closed_jaxpr, name=name)
        
        if isinstance(out_flat, (list, tuple)):
            if len(out_flat) == 1:
                return out_flat[0]
            else:
                return tuple(out_flat)
        else:
            return out_flat
    
    return wrapped