from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing import text, sequence
import pandas as pd
import numpy as np
import re
import unicodedata
from os import path
import pickle

import resources as res
import resources_out as out


class DataProcessor(object):

    def __init__(self, train_d=res.TRAIN_CSV_PATH, test_d=res.TEST_CSV_PATH, clean_text=True, pad_seq=True):
        # type: (str, str, str) -> None
        self._train_d = pd.read_csv(train_d)
        self._test_d = pd.read_csv(test_d)
        # self._test_l = pd.read_csv(test_l)
        self._max_seq_len = 1000
        self._tokenizer = text.Tokenizer(char_level=True, lower=True)  # TODO: max number of words

        self.processed = False  # True after data processing
        self._clean_words = clean_text
        self._pad_seq = pad_seq
        self.classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.seq_train = None  # type: np.ndarray
        self.seq_val = None  # type: np.ndarray
        self.seq_test = None  # type: np.ndarray
        self.labels_train = None  # type: np.ndarray
        self.labels_val = None  # type: np.ndarray

    @staticmethod
    def _clean_text(text):
        formatted = str(unicodedata.normalize('NFKD', text).encode('ascii', 'ignore'))
        exp = re.compile(r'[^(a-z\d\!\@\#\$\%\^\&\*\=\?)*]', re.IGNORECASE)
        clean = exp.sub('', formatted)
        return clean

    def process_data(self):
        text_train = self._train_d["comment_text"].fillna("no comment").values
        text_test = self._test_d["comment_text"].fillna("no comment").values

        if self._clean_words:
            text_train = np.asarray([self._clean_text(t) for t in text_train])
            text_test = np.asarray([self._clean_text(t) for t in text_test])

        print('fitting tokenizer...')
        self._tokenizer.fit_on_texts(texts=list(text_test) + list(text_train))
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

    def dump_seq(self):
        print('saving sequences to: ', out.RES_OUT_DIR)
        np.save(path.join(out.RES_OUT_DIR, 'seq_train.npy'), self.seq_train)
        np.save(path.join(out.RES_OUT_DIR, 'seq_val.npy'), self.seq_val)
        np.save(path.join(out.RES_OUT_DIR, 'seq_test.npy'), self.seq_test)


def example():
    data_pro = DataProcessor()
    data_pro.process_data()
    print('tokens: {}'.format(list(data_pro.get_tokens())))
    print('first sequence: {} \n{}'.format(data_pro.seq_train[0].shape, data_pro.seq_train[0]))
    print('first label: {}'.format(data_pro.labels_train[0]))
    data_pro.dump_seq()


if __name__ == '__main__':
    example()
