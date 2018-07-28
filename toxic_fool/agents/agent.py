from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import tensorflow as tf

import models
import data


class AgentConfig(object):

    def print(self):
        print('|-----------------------------------------|')
        print('|                  CONFIG                 |')
        print('|-----------------------------------------|')
        for k, v in vars(self).items():
            print('|{:25}|{:15}|'.format(k, str(v)))
        print('|-----------------------------------------|')


class Agent(object):

    def __init__(self, sess, tox_model, config):
        # type: (tf.Session, models.ToxicityClassifier, AgentConfig) -> None
        self._sess = sess
        self._config_vars = vars(config)
        self._tox_model = tox_model

    @abc.abstractmethod
    def _build_graph(self):
        pass

    @abc.abstractmethod
    def _train_step(self):
        pass

    @abc.abstractmethod
    def train(self, dataset):
        pass

    @abc.abstractmethod
    def restore(self, restore_path):
        pass

    @abc.abstractmethod
    def attack(self, seq, target_confidence):
        pass
