"""Per-module gradient and actual optimizer-step diagnostics."""
import jax
import jax.numpy as jnp


def module_update_metrics(params, grads, new_params):
    result = {}
    for name in params:
        if not name.startswith('condition_'):
            continue
        def norm(tree):
            return jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32)))
                                for x in jax.tree_util.tree_leaves(tree)))
        delta = jax.tree.map(lambda new, old: new.astype(jnp.float32) - old.astype(jnp.float32),
                             new_params[name], params[name])
        p_norm = norm(params[name])
        result[f'condition_diag/{name}/grad_norm'] = norm(grads[name])
        result[f'condition_diag/{name}/param_norm'] = p_norm
        result[f'condition_diag/{name}/update_norm'] = norm(delta)
        result[f'condition_diag/{name}/relative_update'] = norm(delta) / jnp.maximum(p_norm, 1e-12)
    return result
