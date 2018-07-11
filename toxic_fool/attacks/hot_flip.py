from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from models.toxicity_clasifier_keras import ToxicityClassifierKeras, CalcAccuracy
import data

import numpy as np
import tensorflow as tf
from resources_out import RES_OUT_DIR



if __name__ == '__main__':


    ####################TODO replace all the below code with yotam constractor for the classifer

    # Parse cmd line arguments
    # pylint: disable=C0103

    parser = argparse.ArgumentParser()
    parser.add_argument('-restore_checkpoint', action='store_true', default=False, dest='restore_checkpoint',
                        help='Whether to restore previously saved checkpoint')
    parser.add_argument('-restore_checkpoint_fullpath', action="store", default= RES_OUT_DIR + '/weights.best.hdf5',
                        dest="restore_checkpoint_fullpath", help='Full path of the checkpoint file to restore')
    parser.add_argument('-save_checkpoint', action='store_true', default=False, dest='save_checkpoint',
                        help='Whether to save checkpoints at the end of each epoch')
    parser.add_argument('-save_checkpoint_path', action="store", default=RES_OUT_DIR,
                        dest="save_checkpoint_path", help='Path of the checkpoint directory to save')
    parser.add_argument('-use_gpu', action='store_true', default=False, dest='use_gpu',
                        help='Whether to use gpu')
    parser.add_argument('-recall_weight', action='store',type=float, default=0.001, dest='recall_weight',
                        help='Recall weight in loss function')
    args = parser.parse_args()


    sess = tf.Session()
    embedding_matrix , char_to_token_dic = data.Dataset.init_embedding_from_dump()
    num_tokens, embed_dim = embedding_matrix.shape
    max_seq = 400
    tox_model = ToxicityClassifierKeras(session=sess, max_seq=max_seq, num_tokens=num_tokens, embed_dim=embed_dim,
                                        padded=True, embedding_matrix=embedding_matrix,
                                        metrics=['accuracy', 'ce', CalcAccuracy.precision, CalcAccuracy.recall,
                                                 CalcAccuracy.f1], args=args)

    ##########################################

    #get data
    dataset = data.Dataset.init_from_dump()

    # taking the first sentence. TODO change
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    true_classes = dataset.train_lbl[0, :]

    #squeeze the seq to vector
    squeeze_seq = seq.squeeze(0)

    #print setence before the flip
    tox_model.print_seq(squeeze_seq, char_to_token_dic)

    #copy the sentence to the output sentence
    new_squeeze_seq = squeeze_seq.copy()

    num_of_char_to_flip = 10 #TODO use args

    #loop on the amount of char to flip
    #TODO - if i replace a char twice, count it once
    for k in range(num_of_char_to_flip):
        grad_best_flip = -np.inf

        #get grad
        new_seq = np.expand_dims(new_squeeze_seq, 0)
        grad_tox = tox_model.get_gradient(new_seq)
        char_grad_tox = np.dot(grad_tox, embedding_matrix.T)
        char_grad_tox = np.squeeze(char_grad_tox, axis=0)

        #vairables for the best flip
        index_of_char_to_flip , char_to_check_flip , char_to_flip_to = None , None , None

        #going over all the sentence
        for i in range(tox_model._max_seq):

            curr_char = new_squeeze_seq[i]

            # 0 is special token for nothing , 95 is ' '
            # TODO do generic. TODO maybe allow to replace ' '
            if (curr_char == 0 or curr_char == 95) : continue

            #going over all to possible char to flip to
            for j in range(tox_model._num_tokens):

                curr_flip_grad_diff = char_grad_tox[i][curr_char] - char_grad_tox[i][j]

                #check if this is the best flip
                if curr_flip_grad_diff > grad_best_flip:
                    grad_best_flip = curr_flip_grad_diff
                    index_of_char_to_flip, char_to_check_flip, char_to_flip_to = i , curr_char , j

        #flip char
        print("char number to flip: ", index_of_char_to_flip, "curr char: ",
              char_to_check_flip, "new char:", char_to_flip_to)

        new_squeeze_seq[index_of_char_to_flip] = char_to_flip_to

    #print sentance after the flips
    tox_model.print_seq(new_squeeze_seq, char_to_token_dic)

    #classes before the change
    classes = tox_model.classify(np.expand_dims(squeeze_seq, 0))
    print(classes)

    #classes after the change
    classes = tox_model.classify(np.expand_dims(new_squeeze_seq, 0))
    print(classes)

    print(true_classes)

    # TODO decide which class to attack
    #TODO use beam search

