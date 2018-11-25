from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np
import re
import unicodedata
from os import path

import resources as res
import resources_out as out

SEQ_TRAIN_DUMP = 'seq_train.npy'
SEQ_VAL_DUMP = 'seq_val.npy'
SEQ_TEST_DUMP = 'seq_test.npy'

LBL_TRAIN_DUMP = 'lbl_train.npy'
LBL_VAL_DUMP = 'lbl_val.npy'
LBL_TEST_DUMP = 'lbl_test.npy'

CHAR_EMBEDDING_TEST_DUMP = 'char_embedding.npy'
CHAR_INDEX_TEST_DUMP = 'char_index_embedding.npy'
TRAIN_LABELS_1_RATIO = 'train_labels_1_ratio.npy'


class Dataset(object):
    # pylint: disable=too-many-arguments
    def __init__(self, train_seq, train_lbl, val_seq, val_lbl, test_seq, test_lbl,
                 train_replace_lbl=None, val_replace_lbl=None, test_replace_lbl=None):
        self.train_seq = train_seq
        self.val_seq = val_seq
        self.test_seq = test_seq
        self.train_lbl = train_lbl
        self.val_lbl = val_lbl
        self.test_lbl = test_lbl
        self.train_replace_lbl = train_replace_lbl
        self.val_replace_lbl = val_replace_lbl
        self.test_replace_lbl = test_replace_lbl

    @classmethod
    def init_embedding_from_dump(cls):
        return np.load(path.join(out.RES_OUT_DIR, CHAR_EMBEDDING_TEST_DUMP)), \
               np.load(path.join(out.RES_OUT_DIR, CHAR_INDEX_TEST_DUMP)).item(), \
               np.load(path.join(out.RES_OUT_DIR, TRAIN_LABELS_1_RATIO))

    @classmethod
    def init_from_dump(cls, folder=out.RES_OUT_DIR):
        assert path.isdir(folder), '{} is not a dir'.format(folder)
        train_seq = np.load(path.join(folder, SEQ_TRAIN_DUMP))
        val_seq = np.load(path.join(folder, SEQ_VAL_DUMP))
        test_seq = np.load(path.join(folder, SEQ_TEST_DUMP))
        train_lbl = np.load(path.join(folder, LBL_TRAIN_DUMP))
        val_lbl = np.load(path.join(folder, LBL_VAL_DUMP))

        print('dataset loaded from {}...'.format(folder))
        return cls(train_seq=train_seq, train_lbl=train_lbl, val_seq=val_seq, val_lbl=val_lbl,
                   test_seq=test_seq, test_lbl=None)


class DataProcessor(object):

    def __init__(self, train_d=res.TRAIN_CSV_PATH, test_d=res.TEST_CSV_PATH,
                 clean_text=True, pad_seq=True):
        # type: (str, str, str) -> None
        self._train_d = pd.read_csv(train_d)
        self._test_d = pd.read_csv(test_d)
        # self._test_l = pd.read_csv(test_l)
        self._max_seq_len = 500
        self._max_data_len = 400
        self._tokenizer = text.Tokenizer(char_level=True, lower=False)  # TODO: max number of words

        self.processed = False  # True after data processing
        self._clean_words = clean_text
        self._pad_seq = pad_seq
        self.classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.seq_train = None  # type: np.ndarray
        self.seq_val = None  # type: np.ndarray
        self.seq_test = None  # type: np.ndarray
        self.labels_train = None  # type: np.ndarray
        self.labels_val = None  # type: np.ndarray
        self.labels_test = None  # type: np.ndarray
        self._embedding_matrix = None  # type: np.ndarray
        self._char_index = None


    @staticmethod
    def _clean_text(text_seqs,char_list,max_data_len):

        text_seqs = text_seqs[:max_data_len]

        #replace '\n' (enter) with '. '
        text_seqs = re.sub(r"\n", ". ", text_seqs)

        #remove all char that don't apear in embedding
        chars_in_embedding = ''.join(char_list)
        chars_i_want = set(chars_in_embedding)
        text_seqs = ''.join(c for c in text_seqs if c in chars_i_want)

        return text_seqs + ' ' #i add ' ' at the end for grad calc in case of duplecation of char

    @staticmethod
    def get_char_list_and_embedding_index():
        embeddings_index = {}
        char_data = res.CHAR_EMBEDDING_PATH
        f = open(char_data)
        char_list = []
        for line in f:
            values = line.split()
            curr_char = values[0]
            ##whire space = ' ' - we can't use it in the file, because i use line.split(). so we used 'white_space'
            if curr_char == 'white_space':
                curr_char = ' '

            char_list.append(curr_char)
            value = np.asarray(values[1:], dtype='float32')
            embeddings_index[curr_char] = value
        f.close()

        return char_list, embeddings_index

    # this function was used to create white space embedding once. not needed anymore
    def gen_embedding_for_whitespace(self, embedding_matrix):
        white_space_embedding = np.random.normal(0, 1, [1, 300])  # np.random.rand(1, 300)
        matrix_embedding_norm = np.mean(np.linalg.norm(embedding_matrix, axis=1, keepdims=True))
        white_space_embedding_norm = np.linalg.norm(white_space_embedding, axis=1, keepdims=True)
        white_space_embedding = white_space_embedding / white_space_embedding_norm * matrix_embedding_norm

        return white_space_embedding

    def create_embedding_matrix(self, embeddings_index):
        char_index = self._tokenizer.word_index  # it's actually char and not word. TODO consider fix
        embedding_matrix = np.zeros((len(char_index) + 1, res.EMBEDDING_DIM))
        for char, i in char_index.items():
            embedding_vector = embeddings_index.get(char)
            embedding_matrix[i] = embedding_vector[:res.EMBEDDING_DIM]

        # self.gen_embedding_for_whitespace(embedding_matrix)

        return embedding_matrix

    @staticmethod
    def check_all_data_char_in_embedding(text_train, text_test, embeddings_index):  # TODO move to test
        data_tokanizer = text.Tokenizer(char_level=True, lower=True)
        data_tokanizer.fit_on_texts(texts=list(text_test) + list(text_train))
        char_index = data_tokanizer.word_index
        for char, _ in char_index.items():
            embedding_vector = embeddings_index.get(char)
            if embedding_vector is None:
                raise ValueError('embedding problem, there are char in data which does not exist in embedding.')

    def process_data(self):
        text_train = self._train_d["comment_text"].fillna("no comment").values
        text_test = self._test_d["comment_text"].fillna("no comment").values

        char_list, embedding_index = self.get_char_list_and_embedding_index()

        if self._clean_words:
            text_train = np.asarray([self._clean_text(t,char_list,self._max_data_len) for t in text_train])
            text_test = np.asarray([self._clean_text(t,char_list,self._max_data_len) for t in text_test])

        print('fitting tokenizer...')

        self._tokenizer.fit_on_texts(texts=char_list)
        self._embedding_matrix = self.create_embedding_matrix(embedding_index)
        self.check_all_data_char_in_embedding(text_train, text_test, embedding_index)
        self._char_index = self._tokenizer.word_index

        # self._tokenizer.fit_on_texts(texts=list(text_test) + list(text_train))
        print('done fitting! unique tokens found: {}'.format(len(self._tokenizer.word_index.keys())))

        n_elem = len(text_train)
        np.random.seed(42)
        indices = np.random.permutation(n_elem)
        thresh = n_elem // 10

        val_idx = indices[:thresh]
        train_idx = indices[thresh:]

        labels = self._train_d[self.classes].values
        self.labels_train = list(labels[train_idx])
        self.labels_val = list(labels[val_idx])
        train_labels_cnt = sum(self.labels_train)
        self.train_labels_1_ratio =  train_labels_cnt / len(self.labels_train)
        self.seq_train = self._tokenizer.texts_to_sequences(text_train[train_idx])
        self.seq_val = self._tokenizer.texts_to_sequences(text_train[val_idx])
        self.seq_test = self._tokenizer.texts_to_sequences(text_test)

        if self._pad_seq:
            self.seq_train = sequence.pad_sequences(sequences=self.seq_train, maxlen=self._max_seq_len)
            self.seq_val = sequence.pad_sequences(sequences=self.seq_val, maxlen=self._max_seq_len)
            self.seq_test = sequence.pad_sequences(sequences=self.seq_test, maxlen=self._max_seq_len)

        self.processed = True

        print('processing done! sizes: train {} | val {} | test {}'.format(len(self.seq_train),
                                                                           len(self.seq_val),
                                                                           len(self.seq_test)))

    def get_tokens(self):
        return self._tokenizer.word_index.keys()

    def dump_dataset(self):
        print('saving sequences to: ', out.RES_OUT_DIR)
        np.save(path.join(out.RES_OUT_DIR, SEQ_TRAIN_DUMP), self.seq_train)
        np.save(path.join(out.RES_OUT_DIR, SEQ_VAL_DUMP), self.seq_val)
        np.save(path.join(out.RES_OUT_DIR, SEQ_TEST_DUMP), self.seq_test)
        np.save(path.join(out.RES_OUT_DIR, LBL_TRAIN_DUMP), self.labels_train)
        np.save(path.join(out.RES_OUT_DIR, LBL_VAL_DUMP), self.labels_val)
        # np.save(path.join(out.RES_OUT_DIR, LBL_TEST_DUMP), self.labels_test)
        np.save(path.join(out.RES_OUT_DIR, CHAR_EMBEDDING_TEST_DUMP), self._embedding_matrix)
        np.save(path.join(out.RES_OUT_DIR, CHAR_INDEX_TEST_DUMP), self._char_index)
        np.save(path.join(out.RES_OUT_DIR, TRAIN_LABELS_1_RATIO), self.train_labels_1_ratio)

    def get_dataset(self):
        # type: () -> Dataset
        dataset = Dataset(train_seq=self.seq_train, train_lbl=self.labels_train,
                          val_seq=self.labels_val, val_lbl=self.labels_val,
                          test_seq=self.seq_test, test_lbl=None)
        return dataset


def seq_2_sent(seq, char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    # convert the seq to sentence
    sentance = ''
    for i in range(len(seq)):
        curr_token = seq[i]

        # `0` is a  reserved   index   that won't be assigned to any word.
        if curr_token == 0: continue

        curr_char = token_index[curr_token]
        sentance += curr_char

    return sentance


def example():
    # pylint: disable=unused-variable
    data_pro = DataProcessor()
    data_pro.process_data()
    print('tokens: {}'.format(list(data_pro.get_tokens())))
    print('first sequence: {} \n{}'.format(data_pro.seq_train[0].shape, data_pro.seq_train[0]))
    print('first label: {}'.format(data_pro.labels_train[0]))

    # get dataset
    dataset = data_pro.get_dataset()

    # dump
    data_pro.dump_dataset()

    # load
    dataset_loaded = Dataset.init_from_dump()


if __name__ == '__main__':
    example()
