"""Ape-X DQN Agent Module
"""
from dol.core import Actor, Learner, Builder, ParameterServer
from dol.buffer.transition import TransitionBuffer
from dol.utils import add_batch_dimension, squeeze_batch_dimension
from dol.buffer.adder import SequenceAdder
import numpy as np
import tensorflow as tf
import ray
import time


class ApeXBuilder(Builder):
    def __init__(
        self,
        env_builder,
        network_builder,
        n_actor=7,
        num_advantage=5,
        discount=0.99,
        actor_update_freq=40,
        experience_upload_freq=1000,
        epsilon_base=0.5,
        epsilon_alpha=7,
        batch_size=512,
        learning_rate: float = 0.45,
        lr_decay: float = 0.95,
        rmsprop_epsilon: float = 1.5e-7,
        clipnorm: float = 40,
        update_freq: int = 10,  # update frequence of target network
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
        replay_buffer_size=2e+4

    ):
        ray.init()
        replay_buffer = ray.remote(TransitionBuffer).remote(
            capacity=replay_buffer_size,
            alpha=priority_alpha,
            beta=priority_beta,
            gamma=discount,
            minimum_sample_size=batch_size
        )
        parameter_server = ParameterServer()
        self.learner = ApeXLearner.remote(
            network_builder=network_builder,
            remote_buffer=replay_buffer,
            parameter_server=parameter_server,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_decay=lr_decay,
            rmsprop_epsilon=rmsprop_epsilon,
            discount=discount,
            clipnorm=clipnorm,
            update_freq=update_freq,  # update frequence of target network
            priority_alpha=priority_alpha,
            priority_beta=priority_beta
        )
        self.actors = []
        for i in range(n_actor):
            print(epsilon_base**(1 + i / (n_actor - 1)))
            self.actors.append(ApeXActor.remote(
                env_builder(),
                network_builder,
                replay_buffer,
                parameter_server,
                num_advantage,
                discount,
                actor_update_freq,
                experience_upload_freq,
                epsilon_base**(1 + i / (n_actor - 1))
            ))

    def start(self):
        ids = []
        for actor in self.actors:
            ids.append(actor.run.remote(100000))
        ids.append(self.learner.run.remote())
        ray.wait(ids)


@ray.remote(num_cpus=2)
class ApeXActor(Actor):
    def __init__(
        self,
        env,
        network_builder,
        remote_buffer,
        parameter_server,
        num_advantage=3,
        discount=0.99,
        update_freq=400,  # update frequency to download weights from parameter server
        upload_freq=1000,  # which is half of local memory size
        eps: float = 0.4,  # builder has to calculate agent's eps. See paper

    ):
        # Internalize arguments
        self.env = env
        self.network = network_builder()
        self.remote_buffer = remote_buffer
        self.parameter_server = parameter_server
        self.discount = discount
        self.update_freq = update_freq
        self.upload_freq = 50
        self.eps = eps

        # Initialize states
        self.local_buffer = TransitionBuffer(
            capacity=upload_freq * 2, gamma=discount)
        self.num_action = env.action_spec.num_action
        self.step_count = 0
        self.adder = SequenceAdder(num_advantage, self.local_buffer)

    def get_action(self, envstep):
        if np.random.random() < self.eps:
            action_int = np.random.randint(self.num_action)
        else:
            observation = add_batch_dimension(envstep.observation)
            Q = squeeze_batch_dimension(self.network(observation))
            action_int = tf.argmax(Q).numpy()
        return np.eye(self.num_action)[action_int]

    def update(self):
        weights = self.parameter_server.download_weight()
        if weights:
            self.network.set_weights(weights)

    def on_step_end(self):
        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.update()

        if self.step_count % self.upload_freq == 0:
            sample = self.local_buffer.sample_all()
            priority = self.calc_priority(
                sample.observation,
                sample.action,
                sample.reward,
                sample.next_observation,
                sample.done
            ).numpy()
            self.local_buffer.update_priority(sample.id, priority)
            self.remote_buffer.upload.remote(self.local_buffer)
            self.adder.clear_buffer()

    def on_episode_end(self, step):
        self.adder.add_final_step(step)

    def calc_priority(self, observation, action, reward, next_observation, done):
        # calculate TD Error = priority
        next_action = tf.math.argmax(self.network(
            next_observation), axis=1, output_type=tf.int32)
        tf_range = tf.range(next_action.shape[0], dtype=tf.int32)
        indice = tf.stack([tf_range, next_action], axis=1)
        Q_next = tf.reshape(tf.gather_nd(
            self.network(next_observation), indice), (-1, 1))
        target = reward + Q_next * \
            tf.cast((1 - tf.cast(done, tf.int32)), tf.float32) * self.discount

        Q = tf.math.reduce_max(self.network(
            observation), axis=1, keepdims=True)
        td_error = tf.math.abs(target - Q)

        return tf.squeeze((td_error), axis=1)


@ ray.remote
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
        clipnorm: float = 40,
        update_freq: int = 2500,  # update frequence of target network
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4
    ):
        # internalize arguments
        self.buffer = remote_buffer
        self.update_freq = update_freq
        self.step_count = 0
        self.parameter_server = parameter_server
        self.batch_size = batch_size
        self.clipnorm = clipnorm
        self.discount = discount

        # network
        self.network = network_builder()
        self.network_target = network_builder()
        # optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            epsilon=rmsprop_epsilon,
            centered=True,
            decay=lr_decay,
            clipnorm=clipnorm
        )

    def learn(self):
        self.step_count += 1
        sample = ray.get(self.buffer.sample.remote(self.batch_size))
        if sample is None:
            time.sleep(1)
            return
        else:
            loss, priority = self.learn_from_batch(
                sample.observation,
                sample.action,
                sample.reward,
                sample.next_observation,
                sample.done,
                sample.is_weight
            )
            priority = priority.numpy()
            self.buffer.update_priority.remote(sample.id, priority)

            if self.step_count % self.update_freq == 0:
                self.synchronize_target()
                self.parameter_server.upload_weight(self.network.get_weights())
            if self.step_count % 10 == 0:
                print(loss)

    def learn_from_batch(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        is_weight
    ):
        """Execute single learning process from batch

        Args:
            observation (Union[tf.Tensor, List[tf.Tensor]]): [description]
            action (tf.Tensor): [description]
            reward (tf.Tensor): [description]
            next_observation (Union[tf.Tensor, List[tf.Tensor]]): [description]
            done (tf.Tensor): [description]
            is_weight (tf.Tensor):

        Returns:
            List[tf.Tensor]: list of tensors. loss as the first element and priority as the second.
        """
        # calculate TD Error = priority
        assert len(observation.shape) == 2
        assert len(action.shape) == 2
        assert len(reward.shape) == 2
        assert len(next_observation.shape) == 2
        assert len(done.shape) == 2
        assert len(is_weight.shape) == 2

        next_action = tf.math.argmax(self.network(
            next_observation), axis=1, output_type=tf.int32)
        tf_range = tf.range(next_action.shape[0], dtype=tf.int32)
        indice = tf.stack([tf_range, next_action], axis=1)
        Q_next = tf.reshape(tf.gather_nd(
            self.network(next_observation), indice), (-1, 1))
        target = reward + Q_next * \
            tf.cast((1 - tf.cast(done, tf.int32)), tf.float32) * self.discount

        with tf.GradientTape() as tape:
            Q = tf.math.reduce_max(self.network(
                observation), axis=1, keepdims=True)
            td_error = tf.math.abs(target - Q)
            # calculate loss
            loss = tf.square(td_error) * is_weight
            loss = tf.math.reduce_mean(loss)
            # execute backpropagation
            gradient = tape.gradient(loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradient, self.network.trainable_variables))

        return loss, tf.squeeze(td_error, axis=1)

    def synchronize_target(self):
        self.network_target.set_weights(self.network.get_weights())
