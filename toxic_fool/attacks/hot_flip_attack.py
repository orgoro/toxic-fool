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
import resources as out

HOT_FLIP_ATTACK_TRAIN_FILE =  path.join('data', 'hot_flip_attack_train.npy')
HOT_FLIP_ATTACK_VAL_FILE =  path.join('data', 'hot_flip_attack_val.npy')
HOT_FLIP_ATTACK_TEST_FILE =  path.join('data', 'hot_flip_attack_test.npy')

class HotFlipAttackData(object):
    def __init__(self, hot_flip_status ,sentence_ind):
        self.orig_sent = hot_flip_status.orig_sent
        self.index_of_char_to_flip = hot_flip_status.index_of_char_to_flip
        self.fliped_sent = hot_flip_status.fliped_sent
        self.max_flip_grad_per_char = hot_flip_status.max_flip_grad_per_char
        self.grads_in_fliped_char = hot_flip_status.grads_in_fliped_char
        self.char_to_flip_to = hot_flip_status.char_to_flip_to
        self.sentence_ind = sentence_ind


class HotFlipAttack(object):
    def __init__(self, model, num_of_seq_to_attack= None):
        self.model = model
        self.num_of_seq_to_attack = num_of_seq_to_attack

    def create_data(self, hot_flip_status ,sentence_ind):
        curr_flip_status = hot_flip_status
        sent_attacks = []
        #TODO the data now will be first sentence - last flip
        while curr_flip_status.prev_flip_status != None: ##the original sentence has prev_flip_status = None
            sent_attacks.append(HotFlipAttackData(curr_flip_status, sentence_ind))
            curr_flip_status = curr_flip_status.prev_flip_status

        return sent_attacks

    def save_attack_to_file(self, list_of_hot_flip_attack , file_name):
        np.save(path.join(out.RESOURCES_DIR, file_name), list_of_hot_flip_attack)

    @classmethod
    def load_attack_from_file(self):
        return np.load(path.join(out.RESOURCES_DIR, HOT_FLIP_ATTACK_TRAIN_FILE)),\
               np.load(path.join(out.RESOURCES_DIR, HOT_FLIP_ATTACK_VAL_FILE))
        #np.load(path.join(out.RESOURCES_DIR, HOT_FLIP_ATTACK_TEST_FILE))

    def attack(self,data_seq,labels):

        hot_flip = HotFlip(model=self.model)

        # init list
        list_of_hot_flip_attack = []

        #choosing only the toxic sentences
        index_of_toxic_sent = np.where(labels[:, 0] == 1)[0]

        num_of_seq_to_attack = len(index_of_toxic_sent) if self.num_of_seq_to_attack == None \
                                                        else min( self.num_of_seq_to_attack , len(index_of_toxic_sent))

        #attack first num_of_seq_to_attack sentences
        index_of_toxic_sent = index_of_toxic_sent[: num_of_seq_to_attack]

        t = time.time()

        for i in index_of_toxic_sent:
            seq = np.expand_dims(data_seq[i, :], 0)
            #true_classes = dataset.train_lbl[i, :]

            #do hot flip attack
            best_hot_flip_status , char_to_token_dic = hot_flip.attack(seq = seq )

            #add flip status
            list_of_hot_flip_attack.append( self.create_data(best_hot_flip_status , i) )

            # print sentance after the flips
            print("flipped sentence: ")
            print(data.seq_2_sent(best_hot_flip_status.fliped_sent, char_to_token_dic))

            # # classes before the change
            # print("classes before the flip: ")
            # classes = model.classify(seq)
            # print(classes)
            #
            # # classes after the change
            # print("classes after the flip: ")
            # classes = model.classify(np.expand_dims(best_hot_flip_seq.fliped_sent, 0))
            # print(classes)
            #
            # #print the true class
            # print("true classes: ")
            # print(true_classes)

            dur = time.time() - t
            print("dur is: ", dur)


        return list_of_hot_flip_attack


def example():
    # get restore model
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)

    #create hot flip attack, and attack
    hot_flip_attack = HotFlipAttack(tox_model , num_of_seq_to_attack = 1000) #TODO

    #load dataset
    dataset = data.Dataset.init_from_dump()

    attack_list = []
    attack_list.append((dataset.train_seq, dataset.train_lbl, HOT_FLIP_ATTACK_TRAIN_FILE))
    attack_list.append((dataset.val_seq, dataset.val_lbl, HOT_FLIP_ATTACK_VAL_FILE))
    #attack_list.append((dataset.test_seq, dataset.test_lbl, HOT_FLIP_ATTACK_TEST_FILE))

    for i in range( len(attack_list)):
        seq, label, file_name = attack_list[i]
        #attack this dataset
        list_of_hot_flip_attack = hot_flip_attack.attack(seq,label)
        #save to file
        hot_flip_attack.save_attack_to_file( list_of_hot_flip_attack ,  file_name )

    #load attack data
    loaded_train_hot_flip_attack , _  = hot_flip_attack.load_attack_from_file()

    #the second senetence in data is
    print("seq of the second train sentence: ", loaded_train_hot_flip_attack[1][0].orig_sent )

    #index of char to flip in the first sentence in datatbase, after 1 hot flip
    print("hot flip second flip index of the first sentence", loaded_train_hot_flip_attack[0][1].index_of_char_to_flip)

    #

if __name__ == '__main__':
    example()