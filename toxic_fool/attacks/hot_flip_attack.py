from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import data
import glob
import numpy as np
import tensorflow as tf
from models.toxicity_clasifier_keras import ToxicityClassifierKeras
from attacks.hot_flip import HotFlip
import time
import resources as out

#HOT_FLIP_ATTACK_TRAIN_FILE =  path.join('data', 'hot_flip_attack_train.npy')
#HOT_FLIP_ATTACK_VAL_FILE =  path.join('data', 'hot_flip_attack_val.npy')
#HOT_FLIP_ATTACK_TEST_FILE =  path.join('data', 'hot_flip_attack_test.npy')

class HotFlipAttackData(object):
    def __init__(self, hot_flip_status ,sentence_ind):
        self.orig_sent = hot_flip_status.orig_sent
        self.index_of_char_to_flip = hot_flip_status.index_of_char_to_flip
        self.fliped_sent = hot_flip_status.fliped_sent
        #self.max_flip_grad_per_char = hot_flip_status.max_flip_grad_per_char
        self.grads_in_fliped_char = hot_flip_status.grads_in_fliped_char
        self.char_to_flip_to = hot_flip_status.char_to_flip_to
        self.sentence_ind = sentence_ind


class HotFlipAttack(object):
    def __init__(self, model, num_of_seq_to_attack= None, debug=True, beam_size = 10 , attack_mode = 'dup',
                 stop_after_num_of_flips = True):
        self.model = model
        self.num_of_seq_to_attack = num_of_seq_to_attack
        self.debug=debug
        self.attack_mode = attack_mode
        self.beam_size = beam_size
        self.stop_after_num_of_flips = stop_after_num_of_flips

    def create_data(self, hot_flip_status ,sentence_ind):
        curr_flip_status = hot_flip_status
        sent_attacks = []

        while curr_flip_status.prev_flip_status != None: ##the original sentence has prev_flip_status = None
            sent_attacks.append(HotFlipAttackData(curr_flip_status, sentence_ind))
            curr_flip_status = curr_flip_status.prev_flip_status

        return sent_attacks[::-1] #reverse list, the first flip will be first in list


    def save_attack_to_file(self, list_of_hot_flip_attack , file_name):
        np.save(path.join(out.RESOURCES_DIR, file_name), list_of_hot_flip_attack)

    @classmethod
    def load_attack_from_file(self):

        hot_flip_attack_training = []
        list_of_training_files = glob.glob(path.join(out.RESOURCES_DIR, 'data',
                                                     'split*beam5_hot_flip_attack_train.npy'))
        for training_file in  list_of_training_files:
            loaded_file = np.load(training_file)
            for j in range( len(loaded_file) ):
                hot_flip_attack_training.append(loaded_file[j])

        hot_flip_attack_val = []
        list_of_val_files = glob.glob(path.join(out.RESOURCES_DIR, 'data', 'split*beam10_hot_flip_attack_val.npy'))
        for val_file in  list_of_val_files:
            loaded_file = np.load(val_file)
            for j in range( len(loaded_file) ):
                hot_flip_attack_val.append(loaded_file[j])

        return hot_flip_attack_training,hot_flip_attack_val #\
               #np.load(path.join(out.RESOURCES_DIR, HOT_FLIP_ATTACK_VAL_FILE))
        #np.load(path.join(out.RESOURCES_DIR, HOT_FLIP_ATTACK_TEST_FILE))

    def get_file_name(self,dataset_type,split_num,attack_mode , beam_size):
        initial_file_name = 'split_' + str(split_num) + '_' +str(attack_mode) + '_beam' + str(beam_size) + '_'
        if dataset_type == 'train':
            file_name_to_save = path.join('data', initial_file_name + 'hot_flip_attack_train.npy')
        else:
            file_name_to_save = path.join('data', initial_file_name + 'hot_flip_attack_val.npy')

        return file_name_to_save


    def attack(self,data_seq,labels):

        hot_flip = HotFlip(model=self.model, debug=self.debug, beam_search_size=self.beam_size,
                           attack_mode=self.attack_mode, stop_after_num_of_flips=self.stop_after_num_of_flips)

        # init list
        list_of_hot_flip_attack = []

        #choosing only the toxic sentences
        index_of_toxic_sent = np.where(labels[:, 0] == 1)[0]

        num_of_seq_to_attack = len(index_of_toxic_sent) if self.num_of_seq_to_attack == None \
                                                        else min( self.num_of_seq_to_attack , len(index_of_toxic_sent))

        #attack first num_of_seq_to_attack sentences
        index_of_toxic_sent = index_of_toxic_sent[: num_of_seq_to_attack]

        t = time.time()

        for counter, i in enumerate(index_of_toxic_sent):
            seq = np.expand_dims(data_seq[i, :], 0)
            #true_classes = dataset.train_lbl[i, :]

            #do hot flip attack
            best_hot_flip_status , char_to_token_dic = hot_flip.attack(seq = seq )

            #attack sentence
            curr_hot_flip_attack = self.create_data(best_hot_flip_status , i)

            #add flip status
            if len(curr_hot_flip_attack) > 0:  #if the original sentence was classify below threshold, len = 0
                list_of_hot_flip_attack.append( curr_hot_flip_attack )


            # print sentance after the flips
            if self.debug:
                print("setence num: ", counter ,"flipped sentence: ")
                print(data.seq_2_sent(best_hot_flip_status.fliped_sent, char_to_token_dic))

                dur = time.time() - t
                print("dur is: ", dur)


        return list_of_hot_flip_attack


def example():
    # get restore model
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)

    #create hot flip attack, and attack
    hot_flip_attack = HotFlipAttack(tox_model )

    #load dataset
    dataset = data.Dataset.init_from_dump()

    attack_list = []
    attack_list.append((dataset.train_seq, dataset.train_lbl, 'train'))
    attack_list.append((dataset.val_seq, dataset.val_lbl, 'val'))
    #attack_list.append((dataset.test_seq, dataset.test_lbl, HOT_FLIP_ATTACK_TEST_FILE))

    num_of_split = 10

    for i in range( len(attack_list)):
        seq, label, dataset_type = attack_list[i]

        if len(seq) > 2000 :
            split_seq = np.array_split(seq, num_of_split)
            split_labels = np.array_split(label, num_of_split)
        else:
            split_seq = seq
            split_labels = label

        for j in range ( len(split_seq) ):
            seq_to_attack = split_seq[j]
            label_to_attack = split_labels[j]

            #attack this dataset
            list_of_hot_flip_attack = hot_flip_attack.attack(seq_to_attack,label_to_attack)
            #list_of_hot_flip_attack = []
            #save to file
            file_name_to_save = hot_flip_attack.get_file_name(dataset_type , j,
                                                              hot_flip_attack.attack_mode , hot_flip_attack.beam_size)

            hot_flip_attack.save_attack_to_file( list_of_hot_flip_attack ,  file_name_to_save )

            #to free memmory. i don't think it's really needed
            list_of_hot_flip_attack = None

    #load attack data
    loaded_train_hot_flip_attack , _  = hot_flip_attack.load_attack_from_file()

    #the second senetence in data is
    print("seq of the second train sentence: ", loaded_train_hot_flip_attack[1][0].orig_sent )

    #index of char to flip in the first sentence in datatbase, after 1 hot flip
    print("hot flip second flip index of the first sentence", loaded_train_hot_flip_attack[0][1].index_of_char_to_flip)

    #

if __name__ == '__main__':
    example()
