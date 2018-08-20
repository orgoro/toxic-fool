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
from attacks.hot_flip_attack import HotFlipAttack  ##needed to load hot flip data
from agents.flip_detector import *
from agents.smart_replace import smart_replace
from models import ToxicityClassifierKeras
from models import ToxClassifierKerasConfig
import random


def create_token_dict(char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    return token_index

class RandomFlip(Agent):
    def __init__(self):
        None
    def attack(self, seq, mask, token_index, char_index):
        assert(len(mask) == len(seq))
        masked_seq = seq * mask
        spaces_indices = np.where(seq == 95)
        masked_seq[spaces_indices] = 0
        char_idx_to_flip = random.choice(np.where(masked_seq != 0)[0])
        char_token_to_flip = seq[char_idx_to_flip]
        char_to_flip = token_index[char_token_to_flip]
        char_to_flip_to = smart_replace(char_to_flip)
        token_to_flip_to = char_index[char_to_flip_to]
        flipped_seq = seq
        original_char = seq[char_idx_to_flip]
        flipped_seq[char_idx_to_flip] = token_to_flip_to
        return original_char, char_to_flip_to, char_idx_to_flip, flipped_seq




class AttackerConfig(AgentConfig):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_of_sen_to_attack=100,
                 attack_until_break=True,
                 debug=True):
        self.num_of_sen_to_attack = num_of_sen_to_attack
        self.attack_until_break = attack_until_break
        self.debug = debug
        super(AttackerConfig, self).__init__()



class Attacker():

    def __init__(self,
                 session,
                 tox_model=None,
                 hotflip=None,
                 flip_detector=None,
                 random_flip=None,
                 config=AttackerConfig()):
        self._sess = session
        if flip_detector:
            self._flip_detector = flip_detector
        else:
            flip_detector_config = FlipDetectorConfig(eval_only=True)
            self._flip_detector = FlipDetector(self._sess, config=flip_detector_config)
        if tox_model:
            self._tox_model = tox_model
        else:
            tox_config = ToxClassifierKerasConfig(debug=False)
            self._tox_model  = ToxicityClassifierKeras(self._sess, config=tox_config)
        self._hotflip = hotflip if hotflip else HotFlipAttack(model=self._tox_model, num_of_seq_to_attack=1,debug=False)
        self._random_flip = random_flip if random_flip else RandomFlip()
        self.config = config
        self.dataset = data.Dataset.init_from_dump()
        _, char_index, _ = data.Dataset.init_embedding_from_dump()
        self.char_index = char_index
        self.token_index = create_token_dict(char_index)

    def attack(self,model='random',seq=None, labels=None, mask=None, sequence_idx=0):
        assert (model in ['random', 'hotflip','detector'])
        assert (seq is not None)
        seq = seq.copy()
        curr_seq = seq[sequence_idx]
        if not mask.all():
            mask = np.ones_like(curr_seq)
        sent = data.seq_2_sent(curr_seq, self.char_index)
        tox_before = self._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0]
        if self.config.debug:
            print ("Attacking with model: ", model)
            print ("Toxicity before attack: ", tox_before)
        print(sent)
        if model=='random':
            _,_,flip_idx,res = self._random_flip.attack(curr_seq,mask,self.token_index,self.char_index)
        elif model=='hotflip':
            res = self._hotflip.attack(seq[sequence_idx:],labels)
            flip_idx = res[0][0].char_to_flip_to
            res = res[0][0].fliped_sent
        else:
            _, probs = self._flip_detector.attack(curr_seq, target_confidence=0.)
            mask_probs = probs * mask
            flip_idx = np.argmax(mask_probs, 1)[0]
            token_to_flip = curr_seq[flip_idx]
            char_to_flip = self.token_index[token_to_flip]
            char_to_flip_to = smart_replace(char_to_flip)
            token_of_flip = self.char_index[char_to_flip_to]
            curr_seq[flip_idx] = token_of_flip
            res = curr_seq
        flipped_sent = data.seq_2_sent(res, self.char_index)
        tox_after = self._tox_model.classify(np.expand_dims(res, 0))[0][0]
        if self.config.debug:
            print("Toxicity after attack: ", tox_after)
        print(flipped_sent)
        return tox_before, tox_after, flip_idx, res

    def attack_until_break(self, model='random',seq=None,labels=None,mask=None, sequence_idx=0):
        seq = seq.copy()
        curr_seq = seq[sequence_idx]
        tox = self._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0]
        cnt = 0
        if not mask:
            mask = np.ones_like(curr_seq)
        while tox > 0.5:
            cnt += 1
            _,tox,flip_idx,flipped_seq = self.attack(model=model, seq=seq,labels=labels,mask=mask, sequence_idx=sequence_idx)
            seq[sequence_idx] = flipped_seq
            mask[flip_idx] = 0
        if self.config.debug:
            print ("Toxicity after break: ", tox)
            print ("Number of flips needed: ", cnt)
        return tox, cnt


def example():
    dataset = data.Dataset.init_from_dump()
    sess = tf.Session()
    attacker_config = AttackerConfig(debug=True)
    attacker = Attacker(session=sess, config=attacker_config)

    index_of_toxic_sent = np.where(dataset.val_lbl[:, 0] == 1)[0]

    attack_list = []
    attack_list.append((dataset.val_seq[index_of_toxic_sent], dataset.val_lbl[index_of_toxic_sent], 'val'))

    seq, label, dataset_type = attack_list[0]

    tox_before, tox_after, flip_idx,flipped_seq = attacker.attack(seq=seq, model='random',labels=label)
    tox_before, tox_after, flip_idx,flipped_seq = attacker.attack(seq=seq, model='hotflip',labels=label)
    tox_before, tox_after, flip_idx,flipped_seq = attacker.attack(seq=seq, model='detector',labels=label)
    tox, flips_to_nontoxic =  attacker.attack_until_break(model='detector', seq=seq, labels=label, sequence_idx=10)
    tox, flips_to_nontoxic = attacker.attack_until_break(model='random', seq=seq, labels=label, sequence_idx=10)

if __name__ == '__main__':
    example()
