from chex import assert_trees_all_equal_shapes
import jax
import jax.numpy as jnp
import threading
import builtins


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


class GradientInfoLogger:
    """Thread-safe gradient information logger"""

    def __init__(self):
        # give it a random name
        self.name = f"GradientInfoLogger_{id(self)}"
        self.logged_values = []
        self.lock = threading.Lock()
        self.layer_names = {}

    def log_callback(self, logs):
        with self.lock:
            self.logged_values.append(logs)

    def get_logs_and_clear(self):
        """Get all logged values, optionally clearing the buffer"""
        with self.lock:
            logs = self.logged_values.copy()
            self.logged_values.clear()
            return logs


try:
    _global_logger
except NameError:
    _global_logger = None


def get_logger():
    """Get the global logger instance (process-wide singleton)"""
    global _global_logger
    if getattr(builtins, "_gradient_info_logger", None) is None:
        builtins._gradient_info_logger = _global_logger or GradientInfoLogger()
    _global_logger = builtins._gradient_info_logger
    return _global_logger


def setup_logger():
    """Setup the global logger with specific configuration"""
    global _global_logger
    logger = GradientInfoLogger()
    builtins._gradient_info_logger = logger
    _global_logger = logger
    return logger
