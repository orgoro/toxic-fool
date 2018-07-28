from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data

from models.toxicity_clasifier_keras import ToxicityClassifierKeras, ToxClassifierKerasConfig


def restore():
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)
    return tox_model


def main():
    restore()


if __name__ == '__main__':
    main()
