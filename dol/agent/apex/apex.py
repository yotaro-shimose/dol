"""Ape-X DQN Agent Module
"""
from dol.core import Actor, Learner
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


class ApeXLearner(Learner):
    def __init__(
        self,
        network_builder,
        remote_buffer,
        parameter_server,
        batch_size: int = 512,
        learning_rate: float = 2.5e-4,
        lr_decay: float = 0.95,
        rmsprop_epsilon: float = 1.5e-7,
        discount: float = 0.99,
        norm_clip: float = 40,
        update_freq: int = 2500,  # update frequence of target network
        eps: float = 0.4,  # builder has to calculate agent's eps. See paper.
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4

    ):
        # internalize arguments
        self.buffer = remote_buffer
        self.update_freq = update_freq
        self.step_count = 0
        self.parameter_server = parameter_server
        self.norm_clip = norm_clip
        self.discount = discount

        # network
        self.network = network_builder.create()
        self.network_target = network_builder.create()
        # optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate, epsilon=rmsprop_epsilon, centered=True, decay=lr_decay)

    def learn(self):
        self.step_count += 1
        sample = self.buffer.sample(self.batch_size)
        loss, priority = self.learn_from_batch(
            sample.observation,
            sample.action,
            sample.reward,
            sample.next_observation,
            sample.done,
            sample.is_weight
        )
        self.buffer.update_priority(sample.id, priority)

        if self.step_count % self.update_freq == 0:
            self.synchronize_target()
            self.parameter_server.upload(self.network.get_weights())

    def learn_from_batch(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        is_weight
    ):
        # calculate TD Error = priority
        next_action = tf.math.argmax(self.network(next_observation), axis=1)
        tf_range = tf.range(next_action.shape[0])
        indice = tf.stack([tf_range], next_action)
        Q_next = tf.gather_nd(self.network_target(next_observation), indice)
        target = reward + Q_next * (1 - done) * self.discount

        with tf.GradientTape:
            Q = tf.math.reduce_max(self._network(
                observation), axis=1, keepdims=True)
            td_error = tf.math.abs(target - Q)
            # calculate loss
            loss = tf.square(td_error) * is_weight
            loss = tf.math.reduce_mean(loss)
            # execute backpropagation
            gradient = self.optimizer.get_gradients(
                loss, self.network.trainable_variables)
            # gradients are clipped by norm (maximum norm is 40 in the paper)
            gradient = tf.clip_by_norm(gradient, self.norm_clip)
            self.optimizer.apply_gradients(
                zip(gradient, self.network.trainable_variables))

        return loss, td_error

    def synchronize_target(self):
        self.network_target.set_weights(self.network_get_weights())
