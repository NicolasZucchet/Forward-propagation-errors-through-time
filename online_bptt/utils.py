from chex import assert_trees_all_equal_shapes
import jax
import jax.numpy as jnp


def check_grad_all(grad_1, grad_2, to_check=None, **kwargs):
    # Check that the size matches
    assert_trees_all_equal_shapes(grad_1, grad_2)

    # Create a list of paths to variables to check
    if to_check is not None:
        paths, _ = jax.tree_util.tree_flatten_with_path(to_check)
        paths_to_check = []
        for path in paths:
            paths_to_check.append(
                "/".join([a.key for a in path[0] if a.__class__ == jax.tree_util.DictKey])
                + "/"
                + path[1]
            )
    else:
        flatten_grad_1 = jax.tree_util.tree_flatten_with_path(grad_1)[0]
        paths_to_check = [
            "/".join([a.key for a in g1[0] if a.__class__ == jax.tree_util.DictKey])
            for g1 in flatten_grad_1
        ]

    # For all the parameters to check, verify that they are close to each other
    for path in paths_to_check:
        keys = path.split("/")
        val1, val2 = grad_1, grad_2
        for key in keys:
            val1 = val1[key]
            val2 = val2[key]
        assert jnp.allclose(val1, val2, **kwargs), "Mismatch at %s" % path