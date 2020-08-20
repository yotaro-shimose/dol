import numpy as np
from dol.env.base import ContinuousSpec, DiscreteSpec, TupleSpec


def test_spec():
    low = np.array([0, 3, 2, 5])
    high = np.array([100, 3, 2.5, 20])
    continuous = ContinuousSpec(low, high)
    assert (continuous.low == low).all()
    assert (continuous.high == high).all()
    for _ in range(100):
        assert (low <= continuous.sample()).all()
        assert (continuous.sample() <= high).all()

    n_actions = 4
    discrete = DiscreteSpec(n_actions)
    assert (discrete.low == np.zeros((n_actions,))).all()
    assert (discrete.high == np.ones((n_actions,))).all()
    for _ in range(10):
        assert (discrete.low <= discrete.sample()).all()
        assert (discrete.sample() <= discrete.high).all()

    tup = TupleSpec((continuous, discrete))
    assert (tup.low[0] == low).all()
    assert (tup.high[1] == np.ones((n_actions,))).all()
    assert (tup.shape == ((4,), (4,)))
