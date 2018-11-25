from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import data

import numpy as np
import tensorflow as tf
from models.toxicity_clasifier_keras import ToxicityClassifierKeras
from agents.smart_replace import smart_replace , get_possible_replace

SMALL_LETTERS = 'qwertyuiopasdfghjklzxcvbnm'
CAPITAL_LETTERS = 'QWERTYUIOPASDFGHJKLZXCVBNM'


def create_token_dict(char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    return token_index


class FlipStatus(object):
    #class that hold the curr flip status of the sentence
    #pylint: disable=too-many-arguments
    def __init__(self, fliped_sent, curr_score,index_of_char_to_flip = None,char_to_flip_to = None,orig_sent = None,
                 grads_in_fliped_char = None, max_flip_grad_per_char=None, prev_flip_status = None,mask = None):
        self.fliped_sent = fliped_sent
        self.curr_score = curr_score
        self.index_of_char_to_flip = index_of_char_to_flip
        self.char_to_flip_to = char_to_flip_to
        self.orig_sent = orig_sent
        self.grads_in_fliped_char = grads_in_fliped_char
        self.max_flip_grad_per_char = max_flip_grad_per_char # i think it shouldn't be in use.
        self.prev_flip_status = prev_flip_status
        self.mask = mask

class HotFlip(object):
    # pylint: disable=too-many-arguments
    def __init__(self, model , num_of_char_to_flip = 400, beam_search_size = 5, attack_threshold = 0.10,debug=True,
                 only_smart_replace_allowed = False, replace_only_letters_to_letters = True, attack_mode = 'flip',
                 break_on_half = False, stop_after_num_of_flips = False , num_max_flips = 20):
        self.tox_model = model
        self.num_of_char_to_flip = num_of_char_to_flip
        self.beam_search_size = beam_search_size
        self.attack_threshold = attack_threshold
        self.debug = debug
        self.only_smart_replace_allowed = only_smart_replace_allowed
        self.replace_only_letters_to_letters = replace_only_letters_to_letters
        self.attack_mode = attack_mode
        self.break_on_half = break_on_half
        self.stop_after_num_of_flips = stop_after_num_of_flips
        self.num_max_flips = num_max_flips

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

        best_flip_status = None
        for flip_status in beam_best_flip:
            if flip_status.curr_score > max_score:
                max_score = flip_status.curr_score
                best_flip_status = flip_status

        return best_flip_status

    def create_initial_beam_search_database(self,curr_squeeze_seq,mask):
        beam_best_flip = []
        original_sentence_flip_status = FlipStatus(fliped_sent=curr_squeeze_seq, curr_score=0,mask=mask)
        beam_best_flip.append(original_sentence_flip_status)

        return beam_best_flip

    #get grad for each char in seq
    def get_char_grad_from_seq(self,tox_model , embedding_matrix , squeeze_seq):
        new_seq = np.expand_dims(squeeze_seq, 0)
        grad_tox = tox_model.get_gradient(new_seq)
        char_grad_tox = np.dot(grad_tox, embedding_matrix.T)
        char_grad_tox = np.squeeze(char_grad_tox, axis=0)

        return char_grad_tox

    def unmask_none_latters(self,seq , token_index, mask):

        #convert seq (of char index) to chars
        list_of_chars = [token_index.get(int(char_index), ' ') for char_index in seq.squeeze(0)]

        #check if chars are letters
        list_letter_mask = [(CAPITAL_LETTERS.find(char) >= 0 or SMALL_LETTERS.find(char) >= 0)
                                                                    for char in list_of_chars]

        #muiltiply curr mask with letters mask
        return np.asarray( [int(is_letter) for is_letter in list_letter_mask] ) & mask

    def get_replace_to_mask(self, char,char_to_token_dic,num_tokens):
        replace_to_mask = np.zeros([num_tokens])
        is_capital_letter = CAPITAL_LETTERS.find(char) >= 0
        is_small_letter = SMALL_LETTERS.find(char) >= 0

        if is_capital_letter:
            for char in CAPITAL_LETTERS:
                replace_to_mask[ char_to_token_dic[char] ] = 1

            return replace_to_mask

        if is_small_letter:
            for char in SMALL_LETTERS:
                replace_to_mask[ char_to_token_dic[char] ] = 1

            return replace_to_mask


    def attack(self,seq, mask = None):

        tox_model = self.tox_model

        #get embedding and token dict
        embedding_matrix, char_to_token_dic , _ = data.Dataset.init_embedding_from_dump()

        token_index = create_token_dict(char_to_token_dic)

        # squeeze the seq to vector
        squeeze_seq = seq.squeeze(0)

        if mask is None:
            mask = np.ones_like(squeeze_seq)

        if self.replace_only_letters_to_letters:
            mask = self.unmask_none_latters(seq ,token_index , mask)

        # print sentence before the flip
        if self.debug:
            print( data.seq_2_sent(squeeze_seq, char_to_token_dic) )

        # copy the sentence to the output sentence
        curr_squeeze_seq = squeeze_seq.copy()

        # create initial the beam search database
        beam_best_flip = self.create_initial_beam_search_database(curr_squeeze_seq,mask)


        setence_length_10_per = int(np.sum(curr_squeeze_seq != 0) / 10)

        # loop on the amount of char to flip
        for num_of_flips in range(self.num_of_char_to_flip):

            # get best flip from beam
            best_hot_flip_status = self.get_best_hot_flip(beam_best_flip)
            curr_class = self.tox_model.classify(np.expand_dims(best_hot_flip_status.fliped_sent, 0))[0][0]
            if self.debug:
                print("curr class: ", curr_class)
            if curr_class < 0.5 and num_of_flips >= setence_length_10_per:
                break

            if curr_class < self.attack_threshold:
                break

            if self.break_on_half and curr_class < 0.5:
                break

            if self.stop_after_num_of_flips and num_of_flips > self.num_max_flips:
                break

            #copy the curr database in order not to iterate over the changed database
            copy_beam_best_flip = beam_best_flip.copy()

            for curr_flip in copy_beam_best_flip:

                curr_squeeze_seq = curr_flip.fliped_sent.copy()

                # get grad
                char_grad_tox = self.get_char_grad_from_seq(tox_model, embedding_matrix, curr_squeeze_seq)

                index_of_char_allowed_to_flip = np.argwhere( curr_flip.mask == 1).squeeze(1)

                if self.attack_mode == 'flip':
                    #flip attack
                    max_flip_grad_per_char, flip_grad_matrix = self.flip_grad_calc(index_of_char_allowed_to_flip,
                                                                                   curr_squeeze_seq,
                                                                                   char_grad_tox, tox_model,
                                                                                   char_to_token_dic, token_index)

                    beam_best_flip = self.find_best_flip( tox_model, flip_grad_matrix, beam_best_flip, curr_flip,
                                                         curr_squeeze_seq,  max_flip_grad_per_char)
                else:
                    #dulication attack
                    dup_grad_matrix  = self.duplication_grad_calc(tox_model , curr_squeeze_seq , char_grad_tox)

                    beam_best_flip = self.find_best_duplication_flip(tox_model , dup_grad_matrix , curr_flip ,
                                                                     beam_best_flip , curr_squeeze_seq)



        #get best flip from beam
        best_hot_flip_status  = self.get_best_hot_flip(beam_best_flip)

        return best_hot_flip_status , char_to_token_dic, 

    def find_best_duplication_flip(self, tox_model , dup_grad_matrix , curr_flip , beam_best_flip , curr_squeeze_seq):

        #going over all the sentence for flip
        for i in range(tox_model._max_seq):

            # calc score for curr flip
            curr_flip_grad_diff = dup_grad_matrix[i]

            # calc score for all flip till now
            curr_score = curr_flip.curr_score + curr_flip_grad_diff

            # check if need to update the beam search database with the curr flip
            curr_min_score_in_beam, index = self.get_min_score_in_beam(beam_best_flip)

            if len(beam_best_flip) < self.beam_search_size or curr_score > curr_min_score_in_beam:

                # update beam search database with the new flip
                dup_squeeze_seq = curr_squeeze_seq.copy()
                #first_char_index = np.argmax(dup_squeeze_seq > 0)

                #TODO solve dup of the first char
                if i == 0:
                    continue

                if not i > 0:
                    raise AssertionError()

                #duplicate the i latter
                dup_squeeze_seq[0:i] = dup_squeeze_seq[1:i+1]

                new_flip_status = FlipStatus(fliped_sent=dup_squeeze_seq,
                                             curr_score=curr_score,
                                             index_of_char_to_flip=None, #TODO
                                             char_to_flip_to=None, #TODO
                                             orig_sent=curr_squeeze_seq.copy(),
                                             grads_in_fliped_char=None, #TODO
                                             max_flip_grad_per_char=None, #TODO remove
                                             prev_flip_status=curr_flip,
                                             mask =  curr_flip.mask) #TODO

                if len(beam_best_flip) < self.beam_search_size:
                    beam_best_flip.append(new_flip_status)
                else:
                    beam_best_flip[index] = new_flip_status

        return beam_best_flip

    def duplication_grad_calc(self,tox_model , curr_squeeze_seq , char_grad_tox ):

        #duplicate grad calc
        dup_grad_matrix = np.full((tox_model._max_seq), -np.inf)
        for i in range(tox_model._max_seq):

            curr_index = i
            curr_token = curr_squeeze_seq[curr_index]

            # 0 is special token for nothing , 95 is ' '. # TODO do generic.
            if curr_token == 0 or curr_token == 95: continue

            dup_grad_matrix[i] = 0
            num_of_replace = 0

            #while we didn't reach the end of the word
            while curr_squeeze_seq[curr_index] != 95:

                curr_token = curr_squeeze_seq[curr_index]

                next_char_token = curr_squeeze_seq[curr_index + 1]

                index_of_flip = curr_index + 1
                dup_grad_matrix[i] += char_grad_tox[index_of_flip][next_char_token] \
                                      - char_grad_tox[index_of_flip][curr_token]

                curr_index += 1
                num_of_replace += 1

            #normalization to dup_grad_matrix[i]
            dup_grad_matrix[i] = dup_grad_matrix[i] / np.sqrt( 2 * num_of_replace)

        return dup_grad_matrix



    def flip_grad_calc(self,index_of_char_allowed_to_flip,curr_squeeze_seq , char_grad_tox , tox_model,
                       char_to_token_dic, token_index  ):

        # calc all relevant grads
        flip_grad_matrix = np.full((tox_model._max_seq, tox_model._num_tokens), -np.inf)

        max_flip_grad_per_char = np.full((tox_model._max_seq), -np.inf)

        ##flip grad calc
        for i in index_of_char_allowed_to_flip:

            curr_token = curr_squeeze_seq[i]

            # 0 is special token for nothing , 95 is ' '. # TODO do generic.
            if curr_token == 0 or curr_token == 95: continue

            # if only smart replace allowed, grads in other position will be np.inf
            grad_curr_token = char_grad_tox[i][curr_token]
            if self.only_smart_replace_allowed:
                replace_to_mask = np.zeros([tox_model._num_tokens])
                curr_char = token_index[curr_token]
                pos_flip = get_possible_replace(curr_char)
                token_pos_flip = [char_to_token_dic[word] for word in pos_flip]
                replace_to_mask[token_pos_flip] = 1
                char_grad_tox[i][np.where(replace_to_mask == 0)] = np.inf

            if self.replace_only_letters_to_letters:
                curr_char = token_index[curr_token]
                replace_to_mask = self.get_replace_to_mask(curr_char, char_to_token_dic, tox_model._num_tokens)
                char_grad_tox[i][np.where(replace_to_mask == 0)] = np.inf

            flip_grad_matrix[i] = grad_curr_token - char_grad_tox[i]  # TODO check if it shouldn't be the oposite
            max_flip_grad_per_char[i] = np.max(flip_grad_matrix[i])

        return max_flip_grad_per_char , flip_grad_matrix

    def find_best_flip(self, tox_model , flip_grad_matrix , beam_best_flip , curr_flip , curr_squeeze_seq,
                        max_flip_grad_per_char) :
        # going over all the sentence for flip
        for i in range(tox_model._max_seq):

            # going over all to possible char to flip to
            for j in range(tox_model._num_tokens):

                # calc score for curr flip
                curr_flip_grad_diff = flip_grad_matrix[i][j]  # char_grad_tox[i][curr_char] - char_grad_tox[i][j]

                # calc score for all flip till now
                curr_score = curr_flip.curr_score + curr_flip_grad_diff

                # check if need to update the beam search database with the curr flip
                curr_min_score_in_beam, index = self.get_min_score_in_beam(beam_best_flip)

                if len(beam_best_flip) < self.beam_search_size or curr_score > curr_min_score_in_beam:
                    index_of_char_to_flip, char_to_flip_to = i, j

                    # update beam search database with the new flip
                    fliped_squeeze_seq = curr_squeeze_seq.copy()
                    fliped_squeeze_seq[index_of_char_to_flip] = char_to_flip_to

                    updated_mask = curr_flip.mask.copy()
                    updated_mask[index_of_char_to_flip] = 0

                    new_flip_status = FlipStatus(fliped_sent=fliped_squeeze_seq,
                                                 curr_score=curr_score,
                                                 index_of_char_to_flip=index_of_char_to_flip,
                                                 char_to_flip_to=char_to_flip_to,
                                                 orig_sent=curr_squeeze_seq.copy(),
                                                 grads_in_fliped_char=flip_grad_matrix[i],
                                                 max_flip_grad_per_char=max_flip_grad_per_char,
                                                 prev_flip_status=curr_flip,
                                                 mask = updated_mask)

                    if len(beam_best_flip) < self.beam_search_size:
                        beam_best_flip.append(new_flip_status)
                    else:
                        beam_best_flip[index] = new_flip_status

        return  beam_best_flip


def check_length(dataset):
    my_dataset = dataset.train_seq[np.where(dataset.train_lbl[:, 0] == 1)[0]]
    num_of_sen , sentence_length = my_dataset.shape
    list_of_length = []
    for i in range(num_of_sen):
        j = 0
        while my_dataset[i][j] == 0:
            j = j + 1
        list_of_length.append(sentence_length - j)

    print(list_of_length)



def example():

    # get restore model
    sess = tf.Session()
    tox_model = ToxicityClassifierKeras(session=sess)

    hot_flip = HotFlip(model=tox_model)
    hot_dup = HotFlip(model=tox_model,attack_mode='dup')

    list_of_attack = [ hot_flip , hot_dup ]

    # get data
    dataset = data.Dataset.init_from_dump()

    index_of_toxic_sent = np.where(dataset.train_lbl[:, 0] == 1)[0]

    #check_length(dataset)
    num_of_sentence_to_attack = 5
    for i in range (num_of_sentence_to_attack):

        index_to_attack = index_of_toxic_sent[i]

        # taking the first sentence.
        seq = np.expand_dims(dataset.train_seq[index_to_attack, :], 0)


        for attack in list_of_attack:
            print("attack mode: " , attack.attack_mode)
            #do hot flip attack
            best_flip_status , char_to_token_dic = attack.attack(seq = seq)

            # print sentance after the flips
            print("flipped sentence: ")
            print(data.seq_2_sent(best_flip_status.fliped_sent, char_to_token_dic))

            # classes before the change
            print("tox class before the flip: ")
            classes = tox_model.classify(seq)[0][0]
            print(classes)

            # classes after the change
            print("tox class after the flip: ")
            classes = tox_model.classify(np.expand_dims(best_flip_status.fliped_sent, 0))[0][0]
            print(classes)



        #
        # #print the true class
        # print("true classes: ")
        # print(true_classes)


if __name__ == '__main__':
    example()