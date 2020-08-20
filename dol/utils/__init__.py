import numpy as np


def add_batch_dimension(ndarray):
    ndarray = ndarray.reshape(tuple(1) + ndarray.shape)
    return ndarray


def stack_to_batch(observations):
    # Turn array of tuple of ndarray(stack) into tuple of ndarray with batch dimension.
    # Since tensorflow model's call function accepts inputs as tensor or sequence of tensors,
    # simply stacking tuple of tensors cannot be inputs of the model in case your observations
    # are form of tuple of ndarray.

    # no need to re-construct if observation_space is not tuple.
    if not isinstance(observations[0], tuple):
        return np.array(observations)

    tuple_length = len(observations[0])
    arrays = tuple([] for _ in range(tuple_length))
    for observation in observations:
        for i in range(tuple_length):
            arrays[i].append(observation[i])
    return tuple(np.array(array) for array in arrays)
