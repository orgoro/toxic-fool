from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
from time import gmtime, strftime

import tensorflow as tf
import tqdm
from tensorflow.contrib import slim

import resources_out as res_out
import models
import data
from agents.agent import Agent, AgentConfig


class FlipSelectorConfig(AgentConfig):

    def __init__(self,
                 learning_rate=0.001,
                 training_epochs=100,
                 seq_shape=(None, 400),
                 lbls_shape=(None, 400),
                 batch_size=32,
                 num_units=128,
                 number_of_classes=95,
                 embedding_shape=(96, 400),
                 use_crf=False,
                 training_embed=True):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.seq_shape = seq_shape
        self.lbl_shape = lbls_shape
        self.embedding_shape = embedding_shape
        self.batch_size = batch_size
        self.num_units = num_units  # the number of units in the LSTM cell
        self.number_of_classes = number_of_classes
        self.use_crf = use_crf
        self.train_embed = training_embed
        super(FlipSelectorConfig, self).__init__()


def __str__(self):
    print('________________________')
    for k, v in self.vars.items():
        print('|{:10} | {:10}|'.format(k, v))
    print('________________________')


class FlipSelector(Agent):

    def __init__(self, sess, tox_model=None, config=FlipSelectorConfig()):
        # type: (tf.Session, models.ToxicityClassifier, FlipSelectorConfig) -> None
        self._config = config

        if not tox_model:
            tox_model = models.ToxicityClassifierKeras(sess)
        super(FlipSelector, self).__init__(sess, tox_model, config)

        self._train_op = None
        self._loss = None
        self._probs = None
        self._seq_ph = None
        self._lbl_ph = None
        self._saver = None

        cur_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        self._save_path = path.join(res_out.RES_OUT_DIR, 'flip_selector_' + cur_time)

    def _build_graph(self):
        # inputs
        seq_ph = tf.placeholder(tf.int32, self._config.seq_shape, name="seq_ph")
        lbl_ph = tf.placeholder(tf.float32, self._config.lbl_shape, name="lbl_ph")

        # sizes
        num_units = self._config.num_units
        seq_len = self._config.seq_shape[1]
        num_class = self._config.seq_shape[1]

        embeded = self._embedding_layer(seq_ph)

        # bi-lstm
        # Scope is mandatory to use LSTMCell (https://github.com/tensorflow/tensorflow/issues/799).
        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            _, (state_fwd, state_bwd) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                        cell_bw=lstm_bw_cell,
                                                                        inputs=embeded,
                                                                        sequence_length=seq_len,
                                                                        dtype=tf.float32,
                                                                        scope="BiLSTM")

        # Take only last states
        states = tf.concat([state_fwd, state_bwd], axis=2)
        flat = tf.reshape(states, [-1, 2 * num_units])

        logits = slim.fully_connected(flat, num_class, activation_fn=None)
        probs = tf.sigmoid(logits)

        # Linear-CRF.
        if self._config.use_crf:
            raise NotImplementedError('crf not implemented')
            # log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(probs, lbl_ph, seq_len)
            # # Compute the viterbi sequence and score (used for prediction and test time).
            # viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params,
            #                                                             original_sequence_lengths)

        # Training ops.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=lbl_ph, logits=logits)
        optimizer = tf.train.AdamOptimizer(self._config.learning_rate)
        train_op = optimizer.minimize(loss)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # add entry points
        self._seq_ph = seq_ph
        self._lbl_ph = lbl_ph
        self._probs = probs
        self._train_op = train_op
        self._loss = tf.reduce_sum(loss)
        self._saver = saver

    def _embedding_layer(self, ids):
        vocab_shape = self._config.embedding_shape
        train_embed = self._config.train_embed
        embedding = tf.get_variable('char_embedding', vocab_shape, trainable=train_embed)
        embedded = tf.nn.embedding_lookup(embedding, ids)
        return embedded

    def _train_step(self):
        return self._train_op

    def _print_config(self):
        print('|-----------------------------------------|')
        print('|                  CONFIG                 |')
        print('|-----------------------------------------|')
        for k, v in self._vars.items():
            print('|{:25}|{:15}|'.format(k, v))
        print('|-----------------------------------------|')

    def _get_seq_batch(self, dataset, batch_num, validation=False):
        batch_size = self._config.batch_size
        offset = batch_num * batch_size
        if not validation:
            return dataset.train_seq[offset:offset + batch_size]
        else:
            return dataset.val_seq[offset:offset + batch_size]

    def _get_lbls_batch(self, dataset, batch_num, validation=False):
        batch_size = self._config.batch_size
        offset = batch_num * batch_size
        if not validation:
            return dataset.train_lbl[offset:offset + batch_size]
        else:
            return dataset.val_lbl[offset:offset + batch_size]

    def train(self, dataset):
        self._print_config()
        save_path = self._save_path
        if not path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path.join('model.ckpt')

        num_epochs = self._config.training_epochs
        batch_size = self._config.batch_size
        num_batches = dataset.train_seq.shape[0] // batch_size

        sess = self._sess
        sess.run(tf.global_variables_initializer())

        for e in range(num_epochs):
            seq = self._get_seq_batch(dataset, b, validation=True)
            lbls = self._get_lbls_batch(dataset, b, validation=True)
            feed_dict = {self._seq_ph: seq, self._lbl_ph: lbls}
            fetches = {'loss': self._loss, 'probs': self._probs}
            result = sess.run(fetches, feed_dict)
            print('epoch {:2}/{:2} validation loss: {:5.5} accuracy: XXXX'.format(e, num_epochs, result['loss']))
            self._saver.save(sess, save_path, global_step=e * num_batches)

            p_bar = tqdm.tqdm(range(num_batches))
            for b in p_bar:
                seq = self._get_seq_batch(dataset, b)
                lbls = self._get_lbls_batch(dataset, b)

                feed_dict = {self._seq_ph: seq, self._lbl_ph: lbls}
                fetches = {'loss': self._loss, 'probs': self._probs}

                result = sess.run(fetches, feed_dict)
                p_bar.set_description('epoch {:2}/{:2} | step {:3}/{:3} loss: {:5.5}'.
                                      format(e, num_epochs, b, num_batches, result['loss']))

        sess.close()

    def restore(self, restore_path):
        raise NotImplementedError

    def attack(self, seq, target_confidence):
        raise NotImplementedError


def example():
    dataset = data.Dataset.init_from_dump()
    sess = tf.Session()
    model = FlipSelector(sess)
    model.train(dataset)


if __name__ == '__main__':
    example()