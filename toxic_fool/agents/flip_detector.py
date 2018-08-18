from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
import time
from time import gmtime, strftime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tqdm
import sys

import data
import models
import resources_out as res_out
from agents.agent import Agent, AgentConfig
from data.hot_flip_data_processor import HotFlipDataProcessor
from resources import LATEST_DETECTOR_WEIGHTS
from attacks.hot_flip_attack import HotFlipAttackData  ##needed to load hot flip data


class FlipDetectorConfig(AgentConfig):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 learning_rate=5e-5,
                 training_epochs=100,
                 seq_shape=(None, 400),
                 lbls_shape=(None, 400),
                 batch_size=128,
                 num_units=256,
                 number_of_classes=95,
                 embedding_shape=(96, 300),
                 training_embed=True,
                 num_hidden=1000,
                 restore=True,
                 restore_path=LATEST_DETECTOR_WEIGHTS,
                 eval_only=False,
                 mask_logits=False):
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.seq_shape = seq_shape
        self.lbl_shape = lbls_shape
        self.embedding_shape = embedding_shape
        self.batch_size = batch_size
        self.num_units = num_units  # the number of units in the LSTM cell
        self.number_of_classes = number_of_classes
        self.train_embed = training_embed
        self.num_hidden = num_hidden
        self.restore = restore
        self.restore_path = restore_path
        self.eval_only = eval_only
        self.mask_logits = mask_logits
        super(FlipDetectorConfig, self).__init__()


def __str__(self):
    print('________________________')
    for k, v in self._config_vars.items():
        print('|{:10} | {:10}|'.format(k, v))
    print('________________________')


class FlipDetector(Agent):

    def __init__(self, sess, tox_model=None, config=FlipDetectorConfig()):
        # type: (tf.Session, models.ToxicityClassifier, FlipDetectorConfig) -> None
        self._config = config

        super(FlipDetector, self).__init__(sess, tox_model, config)

        self._train_op = None
        self._summary_op = None
        self._summary_all_op = None
        self._loss = None
        self._val_loss = tf.placeholder(name='val_loss', dtype=tf.float32)
        self._accuracy = tf.placeholder(name='accuracy',dtype=tf.float32)
        self._top5_accuracy = tf.placeholder(name='top5_accuracy',dtype=tf.float32)
        self._probs = None
        self._seq_ph = None
        self._lbl_ph = None

        self._build_graph()
        cur_time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        self._saver = tf.train.Saver()
        self._save_path = path.join(res_out.RES_OUT_DIR, 'flip_detector_' + cur_time)
        if self._config.restore and self._config.eval_only:
            self._sess.run(tf.global_variables_initializer())
            self.restore(self._config.restore_path)

    def build_summary_op(self):
        self._summary_op = tf.summary.merge([
            tf.summary.scalar(name="train_loss", tensor=self._loss)]
        )
        self._summary_all_op = tf.summary.merge([
            tf.summary.scalar(name="val_loss", tensor=self._val_loss),
            tf.summary.scalar(name="accuracy", tensor=self._accuracy),
            tf.summary.scalar(name="top5_accuracy", tensor=self._top5_accuracy)
        ]

        )

    def _build_graph(self):
        # inputs
        seq_ph = tf.placeholder(tf.int32, self._config.seq_shape, name="seq_ph")
        lbl_ph = tf.placeholder(tf.float32, self._config.lbl_shape, name="lbl_ph")
        is_training = tf.placeholder(tf.bool, name='is_training')
        mask_ph = tf.placeholder(tf.float32, self._config.seq_shape, name="mask_ph")


        # sizes
        num_units = self._config.num_units
        num_class = self._config.seq_shape[1]

        embeded = self._embedding_layer(seq_ph)

        # bi-lstm
        # Scope is mandatory to use LSTMCell (https://github.com/tensorflow/tensorflow/issues/799).
        with tf.name_scope("BiLSTM"):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                        cell_bw=lstm_bw_cell,
                                                                        inputs=embeded,
                                                                        dtype=tf.float32,
                                                                        scope="BiLSTM")

        lstm_output = tf.concat((output_fw, output_bw), axis=2)
        flat = tf.reshape(lstm_output, [-1, num_class , 2 * num_units ])

        dropout = tf.layers.dropout(flat, rate = 0.5 * tf.cast(is_training, tf.float32))
        hidden1 = tf.contrib.layers.fully_connected(dropout, 100, activation_fn=tf.nn.relu)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, 50, activation_fn=tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn=None)
        logits = tf.reshape(logits, [-1, num_class])
        masked_logits = tf.multiply(logits, mask_ph)
        probs = tf.nn.softmax(masked_logits)

        # Training ops.
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=lbl_ph, logits=masked_logits)
        optimizer = tf.train.AdamOptimizer(self._config.learning_rate)
        train_op = optimizer.minimize(loss)

        # add entry points
        self._seq_ph = seq_ph
        self._lbl_ph = lbl_ph
        self._mask_ph = mask_ph
        self._is_training = is_training
        self._probs = probs
        self._train_op = train_op
        self._loss = tf.reduce_sum(loss)
        self.build_summary_op()



    def _embedding_layer(self, ids):
        vocab_shape = self._config.embedding_shape
        train_embed = self._config.train_embed
        embedding = tf.get_variable('char_embedding', vocab_shape, trainable=train_embed)
        embedded = tf.nn.embedding_lookup(embedding, ids)
        return embedded

    def _train_step(self):
        return self._train_op

    def _get_seq_batch(self, dataset, batch_num=None, validation=False):
        batch_size = self._config.batch_size
        offset = batch_num * batch_size
        if not validation:
            return dataset.train_seq[offset:offset + batch_size]
        else:
            return dataset.val_seq[offset:offset + batch_size]

    def _get_lbls_batch(self, dataset, batch_num=None, validation=False):
        batch_size = self._config.batch_size
        offset = batch_num * batch_size
        if not validation:
            lbls = dataset.train_lbl[offset:offset + batch_size]
        else:
            lbls = dataset.val_lbl[offset:offset + batch_size]
        lbls_onehot = np.zeros(lbls.shape)
        lbls_onehot[np.arange(batch_size), np.argmax(lbls, axis=1)] = 1
        return  lbls,lbls_onehot

    def _validate(self, dataset):
        batch_size = self._config.batch_size
        num_batches = dataset.val_seq.shape[0] // batch_size
        sess = self._sess
        val_loss = 0
        correct_pred = 0
        correct_top_5_pred = 0
        p_bar = tqdm.tqdm(range(num_batches))
        p_bar.set_description('validation evaluation')
        for batch_num in p_bar:
            seq = self._get_seq_batch(dataset, batch_num, validation=True)
            lbls, lbls_one_hot = self._get_lbls_batch(dataset, batch_num, validation=True)
            if self._config.mask_logits:
                mask = (seq != 0)
            else:
                mask = np.ones_like(seq, dtype=np.float32)
            # evaluate
            feed_dict = {self._seq_ph: seq, self._lbl_ph: lbls_one_hot, self._is_training: False, self._mask_ph: mask }
            fetches = {'loss': self._loss, 'probs': self._probs}
            result = sess.run(fetches, feed_dict)

            # metrics:
            val_loss += result['loss']
            correct_pred += np.sum(np.argmax(lbls, axis=1) == np.argmax(result['probs'], axis=1))
            top_5_label_args = np.argsort(lbls , axis=1)
            for row in range(0, batch_size - 1):
                correct_top_5_pred += np.sum(np.argmax(result['probs'][row]) in top_5_label_args[row, -5:] )
        val_loss = val_loss / dataset.val_seq.shape[0]
        accuracy = correct_pred / dataset.val_seq.shape[0]
        top_5_accuracy = correct_top_5_pred / dataset.val_seq.shape[0]
        if self._config.eval_only:
            print('validation loss: {:5.5} accuracy: {:5.5} top5 accuracy: {:5.5}'.format(
                                                                                        val_loss,
                                                                                        accuracy,
                                                                                        top_5_accuracy))
        return val_loss, accuracy, top_5_accuracy

    def train(self, dataset):
        self._config.print()
        save_path = self._save_path
        if not path.exists(save_path):
            os.mkdir(save_path)
        save_path = path.join(save_path, 'detector_model.ckpt')

        num_epochs = self._config.training_epochs
        batch_size = self._config.batch_size
        num_batches = dataset.train_seq.shape[0] // batch_size

        sess = self._sess
        sess.run(tf.global_variables_initializer())
        if self._config.restore:
            self.restore(self._config.restore_path)
        for e in range(num_epochs):
            summary_writer = tf.summary.FileWriter(self._save_path, flush_secs=30, graph=sess.graph)
            val_loss, accuracy, top5_accuracy = self._validate(dataset)
            sum_tb = self._summary_all_op.eval(session=sess, feed_dict={self._val_loss: val_loss,
                                                                     self._accuracy: accuracy,
                                                                     self._top5_accuracy: top5_accuracy})
            summary_writer.add_summary(sum_tb, e*num_batches)
            time.sleep(0.3)
            print('epoch {:2}/{:2} validation loss: {:5.5} accuracy: {:5.5} top5 accuracy: {:5.5}'.
                  format(e, num_epochs,val_loss,accuracy,top5_accuracy))
            print('saving cheakpoint to: ', save_path)
            time.sleep(0.3)
            self._saver.save(sess, save_path, global_step=e * num_batches)

            p_bar = tqdm.tqdm(range(num_batches))
            for b in p_bar:
                seq = self._get_seq_batch(dataset, b) #TODO i think the dataset is not suffled.
                _, lbls_onehot = self._get_lbls_batch(dataset, b)
                if self._config.mask_logits:
                    mask = (seq != 0)
                else:
                    mask = np.ones_like(seq, dtype=np.float32)
                feed_dict = {self._seq_ph: seq, self._lbl_ph: lbls_onehot, self._is_training: True, self._mask_ph: mask}
                fetches = {'train_op': self._train_op,
                           'loss': self._loss,
                           'probs': self._probs,
                           'sum': self._summary_op}

                result = sess.run(fetches, feed_dict)
                summary_writer.add_summary(result['sum'], e * num_batches + b)
                p_bar.set_description('epoch {:2}/{:2} | step {:3}/{:3} loss: {:5.5}'.
                                      format(e, num_epochs, b, num_batches, result['loss'] / batch_size))

    def restore(self, restore_path):
        # restore:
        saved = self._config.restore_path
        sess = self._sess
        # assert path.exists(saved), 'Saved model was not found'
        self._saver.restore(sess, saved)
        print("Restoring weights from " + saved)

    def attack(self, seq, target_confidence):
        if len(seq.shape) == 1:
            seq = np.expand_dims(seq, 0)
        mask = np.ones_like(seq, dtype=np.float32)
        feed_dict = {self._seq_ph: seq, self._mask_ph: mask}
        result = self._probs.eval(session=self._sess, feed_dict=feed_dict)
        return np.argmax(result, 1) , result


def example():
    dataset = HotFlipDataProcessor.get_detector_datasets()
    _, char_idx, _ = data.Dataset.init_embedding_from_dump()
    sess = tf.Session()
    config = FlipDetectorConfig(restore=True)
    model = FlipDetector(sess, config=config)
    # model._validate(dataset)
    model.train(dataset)

    seq = dataset.train_seq[0]
    flip_idx , _ = model.attack(seq, target_confidence=0.)[0]

    sent = data.seq_2_sent(seq, char_idx)
    flipped_sent = sent[:flip_idx] + '[*]' + sent[min(flip_idx + 1, len(sent)):]
    print(sent)
    print(flipped_sent)
    sess.close()


if __name__ == '__main__':
    example()
