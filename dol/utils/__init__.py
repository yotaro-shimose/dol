import numpy as np
import tensorflow as tf


def stack_to_batch(stack):
    """Turn array of tuple of ndarray(stack) into tuple of ndarray with batch dimension.
    Since tensorflow model's call function accepts inputs as tensor or sequence of tensors,
    simply stacking tuple of tensors cannot be inputs of the model in case your stack
    are form of tuple of ndarray.

    no need to re-construct if observation_space is not tuple.

    Args:
        stack (Sequence): stack = array of tuple of ndarray

    Returns:
        Union[Tuple[tf.Tensor], tf.Tensor]: tuple of Tensor with batch dimension
    """

    if not isinstance(stack[0], tuple):
        return tf.constant(stack)

    stack = tuple(zip(*stack))
    return tuple(tf.stack(observation) for observation in stack)


def add_batch_dimension(value):
    if isinstance(value, tuple):
        return tuple(add_batch_dimension(val) for val in value)
    elif isinstance(value, np.ndarray):
        return value.reshape((1,) + value.shape)
    elif isinstance(value, tf.Tensor):
        return tf.reshape(value, (1,) + value.shape)
    else:
        raise ValueError("Unexpected Value Provided!")
