from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data

import numpy as np
import tensorflow as tf
from models.toxicity_clasifier_keras import ToxicityClassifierKeras
from attacks.hot_flip import HotFlip
from os import path
import time

HOT_FLIP_ATTACK_SAMPLE = 'hot_flip_attack_sample.npy'
HOT_FLIP_OUT_DIR = path.dirname(path.abspath(__file__))

class HotFlipAttackData(object):
    def __init__(self, hot_flip_status ,sentence_ind):
        self.orig_sent = hot_flip_status.orig_sent
        self.index_of_char_to_flip = hot_flip_status.index_of_char_to_flip
        self.fliped_sent = hot_flip_status.fliped_sent
        self.max_flip_grad_per_char = hot_flip_status.max_flip_grad_per_char
        self.grads_in_fliped_char = hot_flip_status.grads_in_fliped_char
        self.char_to_flip_to = hot_flip_status.char_to_flip_to
        self.sentence_ind = sentence_ind



def create_data(hot_flip_status ,sentence_ind):
    curr_flip_status = hot_flip_status
    sent_attacks = []
    while(curr_flip_status.prev_flip_status != None): ##the original sentence has prev_flip_status = None
        sent_attacks.append(HotFlipAttackData(curr_flip_status, sentence_ind))
        curr_flip_status = curr_flip_status.prev_flip_status

    return sent_attacks


def main():

    # get restore model
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)

    hot_flip = HotFlip(model=tox_model)

    # get data
    dataset = data.Dataset.init_from_dump()

    # taking the first sentence.
    num_of_seq_to_attack = 6
    list_of_hot_flip_attack = []

    #choosing only the toxic sentences
    index_of_toxic_sent = np.where(dataset.train_lbl[:, 0] == 1)[0]

    #attack first num_of_seq_to_attack sentences
    index_of_toxic_sent = index_of_toxic_sent[: num_of_seq_to_attack]

    t = time.time()

    for i in index_of_toxic_sent:
        seq = np.expand_dims(dataset.train_seq[i, :], 0)
        true_classes = dataset.train_lbl[i, :]

        #do hot flip attack
        best_hot_flip_seq , char_to_token_dic, flip_status = hot_flip.attack(seq = seq , true_classes = true_classes)

        list_of_hot_flip_attack.append( create_data(flip_status , i) )


        # print sentance after the flips
        print("flipped sentence: ")
        print(data.seq_2_sent(best_hot_flip_seq.fliped_sent, char_to_token_dic))

        if (False):
            # classes before the change
            print("classes before the flip: ")
            classes = tox_model.classify(seq)
            print(classes)

            # classes after the change
            print("classes after the flip: ")
            classes = tox_model.classify(np.expand_dims(best_hot_flip_seq.fliped_sent, 0))
            print(classes)

            #print the true class
            print("true classes: ")
            print(true_classes)

        dur = time.time() - t
        print("dur is: ", dur)


    np.save(path.join(HOT_FLIP_OUT_DIR, HOT_FLIP_ATTACK_SAMPLE), list_of_hot_flip_attack)


if __name__ == '__main__':
    main()


