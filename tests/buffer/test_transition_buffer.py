from dol.buffer.transition import TransitionBuffer
from dol.buffer.adder import SequenceAdder
from dol.core import EnvStep
import numpy as np


def test_reward_calculation():
    n_items = 1000
    n_step = 4
    n_datas = n_items + n_step - 1
    gamma = 0.5
    buffer = TransitionBuffer(n_step, gamma=gamma)
    adder = SequenceAdder(n_step, buffer)

    for _ in range(n_datas):
        obs = (np.random.random((5, 5)), np.zeros(3))
        action = np.random.randint(5)
        reward = 1
        done = False
        step = EnvStep(obs, reward, done)
        adder.add_step(step, action)
    sample = buffer.sample(100)
    assert sample.observation[0].shape == (100, 5, 5)
    assert (sample.reward == sum([(gamma ** i) * 1 for i in range(n_step - 1)])).all()
