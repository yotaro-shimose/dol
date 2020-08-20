"""Ape-X DQN Agent Module
"""
from dol.core import Actor
from dol.buffer.transition import TransitionBuffer
from dol.utils import add_batch_dimension
import numpy as np
import tensorflow as tf


class ApeXActor(Actor):
    def __init__(
        self,
        env,
        adder,
        network_builder,
        remote_buffer,
        num_advantage=3,
        discount=0.99,
        update_freq=400,
        local_memory_size=50,
        eps: float = 0.4,  # builder has to calculate agent's eps. See paper
    ):
        # Internalize arguments
        self.env = env
        self.adder = adder
        self.network = network_builder()
        self.remote_buffer = remote_buffer
        self.discount = discount
        self.update_freq = update_freq
        self.local_memory_size = 50

        # Initialize states
        self.local_buffer = TransitionBuffer(size=local_memory_size)
        self.num_action = env.action_space.n
        self.step_count = 0

    def get_action(self, envstep):
        if np.random.random() < self.eps:
            action = np.random.randint(self.num_action - 1)
        else:
            observation = add_batch_dimension(envstep.observation)
            Q = self.network(observation)
            action = tf.argmax(Q).numpy()
        return action

    def update(self):
        raise NotImplementedError

    def on_step_end(self):
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.update()

        if self.step_count % self.local_memory_size:
            self.remote_buffer.upload(self.local_buffer)
            self.local_buffer.clear()
