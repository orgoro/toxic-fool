from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data

import numpy as np
from models.toxicity_classifier_keras_restore import restore

class FlipStatus(object):
    #class that hold the curr flip status of the sentence
    def __init__(self, fliped_sent, curr_score):
        self.fliped_sent = fliped_sent
        self.curr_score = curr_score

class HotFlip(object):
    def __init__(self, model , num_of_char_to_flip = 4, beam_search_size = 1):
        self.tox_model = model
        self.num_of_char_to_flip = num_of_char_to_flip
        self.beam_search_size = beam_search_size

    #get min score in the beam search
    def get_min_score_in_beam(self, beam_best_flip):
        min_score = np.inf
        i = -1
        for index , flip_status in enumerate(beam_best_flip):
            if flip_status.curr_score < min_score:
                min_score = flip_status.curr_score
                i = index

        return min_score , i

    #get the best setence flip
    def get_best_hot_flip(self,beam_best_flip):
        max_score = -np.inf

        best_flip = None
        for flip_status in beam_best_flip:
            if flip_status.curr_score > max_score:
                max_score = flip_status.curr_score
                best_flip = flip_status

        return best_flip

    def create_initial_beam_search_database(self,curr_squeeze_seq):
        beam_best_flip = []
        original_sentence_flip_status = FlipStatus(fliped_sent=curr_squeeze_seq, curr_score=0)
        beam_best_flip.append(original_sentence_flip_status)

        return beam_best_flip

    #get grad for each char in seq
    def get_char_grad_from_seq(self,tox_model , embedding_matrix , squeeze_seq):
        new_seq = np.expand_dims(squeeze_seq, 0)
        grad_tox = tox_model.get_gradient(new_seq)
        char_grad_tox = np.dot(grad_tox, embedding_matrix.T)
        char_grad_tox = np.squeeze(char_grad_tox, axis=0)

        return char_grad_tox

    def attack(self,seq, true_classes):

        tox_model = self.tox_model

        #get embedding and token dict
        embedding_matrix, char_to_token_dic = data.Dataset.init_embedding_from_dump()

        # squeeze the seq to vector
        squeeze_seq = seq.squeeze(0)

        # print sentence before the flip
        tox_model.print_seq(squeeze_seq, char_to_token_dic)

        # copy the sentence to the output sentence
        curr_squeeze_seq = squeeze_seq.copy()

        # create initial the beam search database
        beam_best_flip = self.create_initial_beam_search_database(curr_squeeze_seq)

        # loop on the amount of char to flip
        # TODO - if i replace a char twice, count it once
        for _ in range(self.num_of_char_to_flip):
            print("flip number: " , _)

            #copy the curr database in order not to iterate over the changed database
            copy_beam_best_flip = beam_best_flip.copy()

            for curr_flip in copy_beam_best_flip:

                curr_squeeze_seq = curr_flip.fliped_sent.copy()

                # get grad
                char_grad_tox = self.get_char_grad_from_seq(tox_model, embedding_matrix, curr_squeeze_seq)

                # going over all the sentence
                for i in range(tox_model._max_seq):

                    curr_char = curr_squeeze_seq[i]

                    # 0 is special token for nothing , 95 is ' '
                    # TODO do generic.
                    # TODO maybe allow to replace ' ' .
                    #TODO maybe replace to other english char
                    if (curr_char == 0 or curr_char == 95): continue

                    # going over all to possible char to flip to
                    for j in range(tox_model._num_tokens):

                        #calc score for curr flip
                        curr_flip_grad_diff = char_grad_tox[i][curr_char] - char_grad_tox[i][j]

                        #calc score for all flip till now
                        curr_score = curr_flip.curr_score + curr_flip_grad_diff

                        #check if need to update the beam search database with the curr flip
                        curr_min_score_in_beam , index = self.get_min_score_in_beam(beam_best_flip)

                        if len(beam_best_flip) < self.beam_search_size or curr_score > curr_min_score_in_beam:
                            index_of_char_to_flip, char_to_flip_to = i, j

                            #update beam search database with the new flip
                            fliped_squeeze_seq = curr_squeeze_seq.copy()
                            fliped_squeeze_seq[index_of_char_to_flip] = char_to_flip_to
                            new_flip_status = FlipStatus(fliped_sent=fliped_squeeze_seq, curr_score=curr_score)

                            if len(beam_best_flip) < self.beam_search_size:
                                beam_best_flip.append(new_flip_status)
                            else:
                                beam_best_flip[index] = new_flip_status


                            # # flip char
                            # print("char number to flip: ", index_of_char_to_flip, "curr char: ",
                            #       char_to_check_flip, "new char:", char_to_flip_to)


        #get best flip from beam
        best_hot_flip_seq = self.get_best_hot_flip(beam_best_flip)

        # print sentance after the flips
        tox_model.print_seq(best_hot_flip_seq.fliped_sent, char_to_token_dic)

        # classes before the change
        classes = tox_model.classify(np.expand_dims(squeeze_seq, 0))
        print(classes)

        # classes after the change
        classes = tox_model.classify(np.expand_dims(best_hot_flip_seq.fliped_sent, 0))
        print(classes)

        print(true_classes)



def example():
    # TODO decide which class to attack

    # get restore model
    tox_model = restore()

    hot_flip = HotFlip(model=tox_model)

    # get data
    dataset = data.Dataset.init_from_dump()

    # taking the first sentence.
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    true_classes = dataset.train_lbl[0, :]

    #do hot flip attack
    hot_flip.attack(seq = seq , true_classes = true_classes)


if __name__ == '__main__':
    example()