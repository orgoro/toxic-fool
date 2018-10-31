from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import data

from models.toxicity_clasifier_keras import ToxicityClassifierKeras, ToxClassifierKerasConfig
from resources_out import RES_OUT_DIR
from resources import LATEST_KERAS_WEIGHTS
import argparse


def train(config):
    sess = tf.Session()
    embedding_matrix = data.Dataset.init_embedding_from_dump()
    max_seq = 500
    config.train_labels_1_ratio = embedding_matrix[2]
    tox_model = ToxicityClassifierKeras(session=sess,
                                        embedding_matrix=embedding_matrix[0],
                                        max_seq=max_seq,
                                        config=config)
    dataset = data.Dataset.init_from_dump()
    tox_model.train(dataset)
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    classes = tox_model.classify(seq)
    print(classes)
    true_classes = dataset.train_lbl[0, :]
    print(true_classes)


def main():
    # Parse cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-restore', action='store_true', default=False, dest='restore',
                        help='Whether to restore previously saved checkpoint')
    parser.add_argument('-restore_path', action="store", default=LATEST_KERAS_WEIGHTS,
                        dest="restore_path", help='Full path of the checkpoint file to restore')
    parser.add_argument('-checkpoint', action='store_true', default=False, dest='checkpoint',
                        help='Whether to save checkpoints at the end of each epoch')
    parser.add_argument('-checkpoint_path', action="store", default=RES_OUT_DIR,
                        dest="checkpoint_path", help='Path of the checkpoint directory to save')
    parser.add_argument('-use_gpu', action='store_true', default=False, dest='use_gpu',
                        help='Whether to use gpu')
    parser.add_argument('-recall_weight', action='store', type=float, default=0.0001, dest='recall_weight',
                        help='Recall weight in loss function')
    parser.add_argument('-run_name', action="store", default='',
                        dest="run_name", help='Will be added to the saved checkpoint names')
    parser.add_argument('-toxic_only', action='store_true', default=False, dest='toxic_only',
                        help='Whether to train on toxic class only')
    args = parser.parse_args()

    config = ToxClassifierKerasConfig(restore=args.restore,
                                      restore_path=args.restore_path,
                                      checkpoint=args.checkpoint,
                                      checkpoint_path=args.checkpoint_path,
                                      # use_gpu=args.use_gpu,
                                      run_name=args.run_name,
                                      train_on_toxic_only=args.toxic_only)

    train(config=config)


if __name__ == '__main__':
    main()
