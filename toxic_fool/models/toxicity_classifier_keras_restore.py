from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data

from models.toxicity_clasifier_keras import ToxicityClassifierKeras, ToxClassifierKerasConfig

def restore():
    config = ToxClassifierKerasConfig(restore=True)
    sess = tf.Session()
    embedding_matrix = data.Dataset.init_embedding_from_dump()
    max_seq = 400
    tox_model = ToxicityClassifierKeras(session=sess, embedding_matrix=embedding_matrix[0], max_seq=max_seq, config=config)
    return tox_model


def main():
    restore()

if __name__ == '__main__':
    main()

