from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data

import numpy as np
from attacks.hot_flip_attack import HotFlipAttack
from attacks.hot_flip_attack import HotFlipAttackData ##needed to load hot flip data

import matplotlib.pyplot as plt #TODO del

def plot_bin(x , file_name):
    plt.bar(range(len(x)), x, align='center', alpha=0.5)
    plt.savefig(file_name)
    plt.close()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class HotFlipDataProcessor(object):

    # def __init__(self, max_seq, num_tokens):
    #     self.max_seq = max_seq
    #     self.num_tokens = num_tokens
    #     #self.hot_flip_attack = hot_flip_attack

    @classmethod
    def extract_flip_data(self, list_of_hot_flip_attack):
        max_seq = len(list_of_hot_flip_attack[0][0].orig_sent)
        num_tokens = len(list_of_hot_flip_attack[0][0].grads_in_fliped_char)

        # calc num of attack sentence
        num_of_sentence_flip_in_database = np.sum([len(sentences) for sentences in list_of_hot_flip_attack])

        # init database
        predections_detector = np.zeros([num_of_sentence_flip_in_database, max_seq])
        sentence_token_input = np.zeros([num_of_sentence_flip_in_database, max_seq])
        #predections_char_selector = None
        predections_char_selector = np.zeros([num_of_sentence_flip_in_database, num_tokens])
        char_detector_index = np.zeros([num_of_sentence_flip_in_database])

        # update database from hot flip attack
        index = 0
        for sentence_attack in list_of_hot_flip_attack:
            for sentence in sentence_attack:
                sentence_token_input[index] = sentence.orig_sent
                predections_detector[index][sentence.index_of_char_to_flip] = 1
                #predections_detector[index] = softmax(sentence.max_flip_grad_per_char)
                predections_char_selector[index][sentence.char_to_flip_to] = 1
                char_detector_index[index] = sentence.index_of_char_to_flip
                index += 1

        return sentence_token_input, predections_detector, predections_char_selector

    @classmethod
    def get_hot_flip_data(self):

        # load hot flip attack
        # list_of_hot_flip_attack_train, list_of_hot_flip_attack_val  = HotFlipAttack.load_attack_from_file()
        list_of_hot_flip_attack_train, _ = HotFlipAttack.load_attack_from_file()
        train_token_input, train_predections_detector, train_predections_selector = \
            self.extract_flip_data(list_of_hot_flip_attack_train)

        # val_token_input, val_predections_detector, val_predections_char_selector = \
        #     self.extract_flip_data(list_of_hot_flip_attack_val)

        np.random.seed(42)
        # p = np.random.permutation(len(train_token_input))
        # train_token_input = train_token_input[p]
        # train_predections_detector = train_predections_detector[p]
        # if train_predections_char_selector != None: #TODO - Verify it's not None
        #     train_predections_char_selector = train_predections_char_selector[p]
        val_size = 10000
        return train_token_input[:-val_size],\
               train_predections_detector[:-val_size], \
               train_predections_selector[:-val_size], \
               train_token_input[-val_size:],\
               train_predections_detector[-val_size:], \
               train_predections_selector[-val_size:]


    @classmethod
    def get_detector_datasets(self):
        train_token_input, train_predections_detector, _, \
        val_token_input, val_predections_detector, _ = self.get_hot_flip_data()

        detector_dataset = data.Dataset(train_seq = train_token_input, train_lbl = train_predections_detector,
                            val_seq = val_token_input, val_lbl = val_predections_detector,
                                        test_seq = None, test_lbl = None) #TODO test

        return detector_dataset

    @classmethod
    def get_char_selector_datasets(self):
        train_token_input, _, train_predections_char_selector, \
        val_token_input, _, val_predections_char_selector = self.get_hot_flip_data()

        char_selector_dataset = data.Dataset(train_seq = train_token_input, train_lbl = train_predections_char_selector,
                            val_seq = val_token_input, val_lbl = val_predections_char_selector,
                                             test_seq = None, test_lbl = None) #TODO test

        return char_selector_dataset

    @classmethod
    def get_detector_selector_datasets(self):
        train_token_input, train_predections_detector, train_predections_char_selector, \
        val_token_input, val_predections_detector, val_predections_char_selector = self.get_hot_flip_data()

        char_selector_dataset = data.Dataset(train_seq=train_token_input,
                                             train_lbl=train_predections_detector,
                                             train_replace_lbl=train_predections_char_selector,
                                             val_seq=val_token_input, val_lbl=val_predections_detector,
                                             val_replace_lbl=val_predections_char_selector,
                                             test_seq=None, test_lbl=None, test_replace_lbl=None)  # TODO test

        return char_selector_dataset

def example():

    # get hot flip data
    dataset = HotFlipDataProcessor.get_detector_selector_datasets()
    # get embedding and token dict
    _, char_to_token_dic, _ = data.Dataset.init_embedding_from_dump()

    print("input stentence 0: ")
    print(data.seq_2_sent(dataset.train_seq[0], char_to_token_dic))

    print("prediction detector: ")
    print(dataset.train_lbl[0])

    print("prediction char selector: ")
    print(dataset.train_replace_lbl[0])

    print("index of the char the should be fliped in the sentence 5")
    print(np.where(dataset.train_lbl == 1)[1][5])

    print("index of the char to flip to in the sentence 5")
    print(np.where(dataset.train_replace_lbl == 1)[1][5])

    # for i in range (4):
    #     file_name = 'flip_' + str(i) + '.png'
    #     plot_bin(predections_char_selector[i][int(char_detector_index[i])],file_name )

    # for i in range (12):
    #     print(data.seq_2_sent(sentence_token_input[i], char_to_token_dic))
    #     file_name = 'flip_' + str(i) + '.png'
    #     plot_bin(predections_detector[i],file_name )


if __name__ == '__main__':
    example()