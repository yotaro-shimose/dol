import tensorflow as tf
import numpy as np


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
        return cast32(tf.stack(stack))

    stack = tuple(zip(*stack))
    return cast32(tuple(tf.stack(observation) for observation in stack))


def cast32(target):
    """Cast dtype of float/int tensor to float/int32.
    Since Tensorflow cannot specify default dtype of entire framework, replay buffer force every
    tensor to be of dtype xxx32 by calling this function.
    Input tensor "target" can be tensor or tuple of tensor.

    Raises:
        TypeError: only accept tf.tensor or tuple of them.

    Args:
        target(Union[Tuple[tf.Tensor], tf.Tensor]): Either tensor or tuple of tensors
    """

    if isinstance(target, tuple):
        return tuple(cast32(value) for value in target)
    elif isinstance(target, tf.Tensor):
        if target.dtype == tf.bool:
            return target
        elif target.dtype == tf.float64 or target.dtype == tf.float16:
            return tf.cast(target, tf.float32)
        elif target.dtype == tf.int64 or target.dtype == tf.int16:
            return tf.cast(target, tf.int32)
        else:
            return target
    else:
        raise TypeError(f"Unexpected input type: {type(target)}")


def add_batch_dimension(value):
    if isinstance(value, tuple):
        return tuple(add_batch_dimension(val) for val in value)
    elif isinstance(value, np.ndarray):
        return value.reshape((1,) + value.shape)
    elif isinstance(value, tf.Tensor):
        return tf.reshape(value, (1,) + value.shape)
    else:
        raise ValueError("Unexpected Observation Provided!")


def squeeze_batch_dimension(value):
    return tf.squeeze(value, axis=0)
