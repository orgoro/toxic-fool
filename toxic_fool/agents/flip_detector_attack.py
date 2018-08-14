
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import time
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tqdm
import sys

import data
import models
from models.toxicity_clasifier_keras import ToxicityClassifierKeras
import resources_out as res_out
from agents.agent import Agent, AgentConfig
from data.hot_flip_data_processor import HotFlipDataProcessor
from resources import LATEST_DETECTOR_WEIGHTS
from attacks.hot_flip_attack import HotFlipAttackData  ##needed to load hot flip data
from agents.flip_detector import *


def example():

    # get restore model
    #sess = tf.Session()

    dataset = data.Dataset.init_from_dump()
    #dataset = HotFlipDataProcessor.get_detector_datasets()
    _, char_idx, _ = data.Dataset.init_embedding_from_dump()
    sess = tf.Session()
    config = FlipDetectorConfig(eval_only=True)
    model = FlipDetector(sess, config=config)
    # model._validate(dataset)

    index_of_toxic_sent = np.where(dataset.val_lbl[:, 0] == 1)[0]

    seq = dataset.val_seq[index_of_toxic_sent[0]]
    sent = data.seq_2_sent(seq, char_idx)

    print(sent)
    tox_model = ToxicityClassifierKeras(session=sess)
    print('toxic class before: ', tox_model.classify(np.expand_dims(seq, 0))[0][0])

    flipped_seq = seq.copy()
    token_to_flip = char_idx['^']
    for i in range(3):
        flip_idx = model.attack(flipped_seq, target_confidence=0.)[0]
        flipped_seq[flip_idx] = token_to_flip

        print(data.seq_2_sent(flipped_seq, char_idx))
        print('toxic class after: ', tox_model.classify(np.expand_dims(flipped_seq, 0))[0][0])

    sess.close()


if __name__ == '__main__':
    example()
