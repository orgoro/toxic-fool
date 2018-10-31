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
from agents.flip_detector import FlipDetector, FlipDetectorConfig
from agents.smart_replace import smart_replace

def get_callsifier(model,seq):
    return model.classify(np.expand_dims(seq, 0))[0][0]

def create_token_dict(char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    return token_index

def example():

    # get restore model

    dataset = data.Dataset.init_from_dump()

    _, char_idx, _ = data.Dataset.init_embedding_from_dump()
    token_index = create_token_dict(char_idx)
    sess = tf.Session()
    config = FlipDetectorConfig(eval_only=True)
    model = FlipDetector(sess, config=config)


    index_of_toxic_sent = np.where(dataset.val_lbl[:, 0] == 1)[0]
    tox_model = ToxicityClassifierKeras(session=sess)
    num_of_sentence_to_attack = 100 #TODO
    for j in range(num_of_sentence_to_attack):
        seq = dataset.val_seq[index_of_toxic_sent[j]]
        sent = data.seq_2_sent(seq, char_idx)

        print(sent)
        flipped_seq = seq.copy()
        curr_class = get_callsifier(tox_model, flipped_seq)

        print('toxic class before: ', curr_class )

        #token_to_flip = char_idx['^']
        #for i in range(3):
        mask_char_allow_to_flip = np.ones([500])
        num_of_flips_done = 0
        while curr_class > 0.5 and num_of_flips_done < 15:
            _ , probs = model.attack(flipped_seq, target_confidence=0.)
            mask_probs = probs * mask_char_allow_to_flip
            flip_idx = np.argmax(mask_probs, 1)[0]
            mask_char_allow_to_flip[flip_idx] = 0
            #curr_sentence = data.seq_2_sent(flipped_seq, char_idx)
            token_to_flip = flipped_seq[flip_idx]
            char_to_flip = token_index[token_to_flip]
            char_to_flip_to = smart_replace(char_to_flip)
            token_of_flip = char_idx[char_to_flip_to]
            flipped_seq[flip_idx] = token_of_flip
            curr_class = get_callsifier(tox_model, flipped_seq)

            print(data.seq_2_sent(flipped_seq, char_idx))
            print('char index that was flipped' , flip_idx)
            print('toxic class after: ', curr_class)
            num_of_flips_done += 1

        print('done attacking sentence')

    sess.close()


if __name__ == '__main__':
    example()