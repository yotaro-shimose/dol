import copy
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from dol.core import Item, StepData


class Adder(ABC):
    """Adder will interact with ReplayBuffer's low level interface which enables workers to
    call adder.add_step(step, action) without recognization of detailed ReplayBuffer implementation.
    Note that Adder DOES NOT calculate priorities! You still have to initialize priority
    by calling buffer.update_priority().

    Extend this class to make algorithm-specific insert strategy.
    """

    @abstractmethod
    def add_step(self, step, action, extra={}):
        """Automatically insert appropriate form of datas by using step and action information.

        Args:
            step (Step): single step object
            action (Action): single action object
        """
        raise NotImplementedError

    def _get_dummy(self, element):
        """create 0-padding dummy element correspoinding to an element shape.

        Args:
            element (Any): element of data like observation, action, reward or extra.
        """
        if isinstance(element, np.ndarray):
            return np.zeros(element.shape)
        elif isinstance(element, tuple):
            return tuple(self._get_dummy(ele) for ele in element)
        elif isinstance(element, int) or isinstance(element, float):
            return 0
        elif isinstance(element, dict):
            return {key: self._get_dummy(value) for key, value in element.items()}

    def _calculate_padding_data(self, data):
        """create 0-padding dummy data corresponding to first data given to this adder.
        override this method when custom Data class is used.

        Args:
            data (Data): data object which is given to this buffer for the first time.
        """
        vars_dict = vars(data)
        self._padding_data = \
            StepData(**{key: self._get_dummy(value)
                        for key, value in vars_dict.items()})


class SequenceAdder(Adder):
    """Insert step and action to LOCAL memory every action step.
    """

    def __init__(self, n_step, buffer):
        """create N-Step Sequence Adder which insert a sequence of datas.
        When provided n == 2, this will simply create transitions in local buffer.

        Args:
            n_step ([type]): length of sequence
            buffer (ReplayBuffer): local replay buffer to insert datas into.
        """
        self._n_step = n_step
        self._buffer = buffer
        self._ids = deque(maxlen=n_step)

    def add_final_step(self, step, extra: dict = {}):
        """add final step with dummy action and padding data if needed.
        always call this function at the end of episode.
        passing step with done = True DOES NOT execute the on_step_end process.

        Args:
            step (Step): step
            extra (dict, optional): extra info. see Data class docstring
        """
        dummy_action = copy.deepcopy(self._padding_data.action)
        final_data = StepData(step.obs, dummy_action,
                              step.reward, step.done, extra)
        _id = self._buffer.add_data(final_data)
        self._ids.append(_id)
        if len(self._ids) == self._n_step:
            item = Item(list(self._ids), 1)
            self._buffer.add_item(item)
        # execute 0-padding
        if self._n_step > 2:
            for _ in range(self._n_step - 2):
                data = copy.deepcopy(self._padding_data)
                _id = self._buffer.add_data(data)
                self._ids.append(_id)
                if len(self._ids) == self._n_step:
                    item = Item(list(self._ids), 1)
                    self._buffer.add_item(item)
        # initialize ids
        self._ids = deque(maxlen=self._n_step)

    def add_step(self, step, action, extra: dict = {}):
        """calling add_step automatically stores informations and create appropriate datas and items
        to create consecutive sequence.

        Args:
            step (Step): step
            action (Action): action
            extra (dict, optional): extra info. see Data class Docstring
        """
        data = StepData(step.observation, action,
                        step.reward, step.done, extra)
        # calculate dummy data for padding when first data is observed
        if len(self._ids) == 0:
            self._calculate_padding_data(data)
        _id = self._buffer.add_data(data)
        self._ids.append(_id)
        if len(self._ids) == self._n_step:
            item = Item(list(self._ids), 1)
            self._buffer.add_item(item)
