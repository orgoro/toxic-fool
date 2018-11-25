from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from resources_out import RES_OUT_DIR
from attacks.hot_flip_attack import HotFlipAttackData  ##needed to load hot flip data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class FlipDetectorConfig(AgentConfig):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 learning_rate=5e-5,
                 training_epochs=1000,
                 seq_shape=(None, 500),
                 lbls_shape=(None, 500),
                 replace_chars_shape=(None, 96),
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
        self.replace_chars_shape = replace_chars_shape
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
        self._accuracy = tf.placeholder(name='accuracy', dtype=tf.float32)
        self._accuracy_select = tf.placeholder(name='accuracy', dtype=tf.float32)
        self._top5_accuracy = tf.placeholder(name='top5_accuracy', dtype=tf.float32)
        self._top5_select_accuracy = tf.placeholder(name='top5_accuracy', dtype=tf.float32)
        self._detect_probs = None
        self._select_probs = None
        self._seq_ph = None
        self._lbl_ph = None
        self._replace_chars_ph = None

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
            tf.summary.scalar(name="accuracy_select", tensor=self._accuracy_select),
            tf.summary.scalar(name="top5_accuracy", tensor=self._top5_accuracy),
            tf.summary.scalar(name="top5_accuracy_select", tensor=self._top5_select_accuracy),

        ]

        )

    def _build_graph(self):
        # inputs
        seq_ph = tf.placeholder(tf.int32, self._config.seq_shape, name="seq_ph")
        lbl_ph = tf.placeholder(tf.float32, self._config.lbl_shape, name="lbl_ph")
        replace_chars_ph = tf.placeholder(tf.float32, self._config.replace_chars_shape, name="replace_chars_ph")
        is_training = tf.placeholder(tf.bool, name='is_training')
        mask_ph = tf.placeholder(tf.float32, self._config.seq_shape, name="mask_ph")

        # sizes
        num_units = self._config.num_units
        num_class = self._config.seq_shape[1]
        num_chars = self._config.embedding_shape[0]
        batch_size = self._config.batch_size

        embeded = self._embedding_layer(seq_ph)
        state_vec = self._build_lstm(embeded, num_class, num_units)
        dropout = tf.layers.dropout(state_vec, rate=0.5 * tf.cast(is_training, tf.float32))

        with tf.variable_scope('flip_detector'):
            detect_masked_logits, detect_probs = self._detect(dropout, num_class, mask_ph)
            detect_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=lbl_ph,
                                                                     logits=detect_masked_logits)
        with tf.variable_scope('flip_selector'):
            idx = tf.convert_to_tensor(np.arange(batch_size)[:, None])
            correct_char = tf.argmax(lbl_ph, axis=1)[:, None]
            chars_idx = tf.concat((idx, correct_char), axis=1)
            select_masked_logits, select_probs = self._select(dropout, num_chars, chars=chars_idx)
            select_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=replace_chars_ph,
                                                                     logits=select_masked_logits)
        combined_loss = tf.reduce_sum(detect_loss) + tf.reduce_sum(select_loss)
        optimizer = tf.train.AdamOptimizer(self._config.learning_rate)
        train_op = optimizer.minimize(detect_loss + select_loss)

        # add entry points
        self._seq_ph = seq_ph
        self._lbl_ph = lbl_ph
        self._replace_chars_ph = replace_chars_ph
        self._mask_ph = mask_ph
        self._is_training = is_training
        self._detect_probs = detect_probs
        self._select_probs = select_probs
        self._train_op = train_op
        self._loss = tf.reduce_sum(combined_loss)
        self.build_summary_op()

    def _build_lstm(self, embeded, num_class, num_units):
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
        state_vec = tf.reshape(lstm_output, [-1, num_class, 2 * num_units])
        return state_vec

    def _detect(self, dropout, num_class, mask=None):
        hidden1 = tf.contrib.layers.fully_connected(dropout, 100, activation_fn=tf.nn.relu)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, 50, activation_fn=tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(hidden2, 1, activation_fn=None)
        logits = tf.reshape(logits, [-1, num_class])
        masked_logits = tf.multiply(logits, mask)
        probs = tf.nn.softmax(masked_logits)
        return masked_logits, probs

    def _select(self, dropout, num_class, chars):
        correct_char_path = tf.gather_nd(dropout, chars)
        hidden1 = tf.contrib.layers.fully_connected(correct_char_path, 100, activation_fn=tf.nn.relu)
        hidden2 = tf.contrib.layers.fully_connected(hidden1, 100, activation_fn=tf.nn.relu)
        logits = tf.contrib.layers.fully_connected(hidden2, num_class, activation_fn=None)
        logits = tf.reshape(logits, [-1, num_class])
        probs = tf.nn.softmax(logits)
        return logits, probs

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
        lbls_onehot = lbls #Lables is already one hot
        return lbls, lbls_onehot

    def _get_replace_batch(self, dataset, batch_num=None, validation=False):
        # pylint: disable=unused-variable
        batch_size = self._config.batch_size
        offset = batch_num * batch_size
        dataset = dataset

        if not validation:
            replace_chars_onehot = dataset.train_replace_lbl[offset: offset+batch_size]
        else:
            replace_chars_onehot = dataset.val_replace_lbl[offset: offset + batch_size]
        replace_chars = np.where(replace_chars_onehot==1)[1]
        return replace_chars, replace_chars_onehot

    def _validate(self, dataset):
        batch_size = self._config.batch_size
        num_batches = dataset.val_seq.shape[0] // batch_size
        sess = self._sess
        val_loss = 0
        correct_pred = 0
        correct_select_pred = 0
        correct_top_5_pred = 0
        correct_top_5_select_pred = 0
        p_bar = tqdm.tqdm(range(num_batches))
        p_bar.set_description('validation evaluation')
        for batch_num in p_bar:
            is_validate = True
            feed_dict, lbls, replace_char = self._get_feed_dict(batch_num, dataset, is_validate)
            fetches = {'loss': self._loss, 'probs': self._detect_probs, 'select_probs': self._select_probs}
            result = sess.run(fetches, feed_dict)

            # metrics:
            val_loss += result['loss']
            correct_pred += np.sum(np.argmax(lbls, axis=1) == np.argmax(result['probs'], axis=1))
            correct_select_pred += np.sum(np.argmax(replace_char,axis=1) == np.argmax(result['select_probs'], axis=1))
            top_5_probs = np.argsort(result['probs'], axis=1)
            top_5_select_probs = np.argsort(result['select_probs'], axis=1)
            for row in range(0, batch_size - 1):
                correct_top_5_pred += np.sum(np.argmax(lbls[row]) in top_5_probs[row, -5:])
                correct_top_5_select_pred += np.sum(np.argmax(replace_char[row]) in top_5_select_probs[row, -5:])
        val_loss = val_loss / (batch_size*num_batches)
        accuracy = correct_pred / (batch_size*num_batches)
        accuracy_select = correct_select_pred / (batch_size*num_batches)
        top_5_accuracy = correct_top_5_pred / (batch_size*num_batches)
        top_5_select_accuracy = correct_top_5_select_pred / (batch_size*num_batches)
        if self._config.eval_only:
            print(
                'validation loss: {:5.5} '
                'accuracy: {:5.5} accuracy_select: {:5.5} '
                'top5 accuracy: {:5.5}  '
                'top5 select accuracy: {:5.5}'.format(
                val_loss,
                accuracy,
                accuracy_select,
                top_5_accuracy,
                top_5_select_accuracy))
        return val_loss, accuracy, top_5_accuracy, accuracy_select, top_5_select_accuracy

    def _get_feed_dict(self, batch_num, dataset, is_validate):
        seq = self._get_seq_batch(dataset, batch_num, validation=is_validate)
        lbls, lbls_one_hot = self._get_lbls_batch(dataset, batch_num, validation=is_validate)
        _, replace_chars_one_hot = self._get_replace_batch(dataset, batch_num, validation=is_validate)
        if self._config.mask_logits:
            mask = (seq != 0)
        else:
            mask = np.ones_like(seq, dtype=np.float32)
        # evaluate
        feed_dict = {self._seq_ph: seq,
                     self._lbl_ph: lbls_one_hot,
                     self._replace_chars_ph: replace_chars_one_hot,
                     self._is_training: is_validate,
                     self._mask_ph: mask}
        return feed_dict, lbls, replace_chars_one_hot

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
            val_loss, accuracy, top5_accuracy, accuracy_select, top5_select_accuracy = self._validate(dataset)
            sum_tb = self._summary_all_op.eval(
                session=sess, feed_dict={self._val_loss: val_loss,
                                         self._accuracy: accuracy,
                                         self._accuracy_select: accuracy_select,
                                         self._top5_accuracy: top5_accuracy,
                                         self._top5_select_accuracy: top5_select_accuracy})
            summary_writer.add_summary(sum_tb, e * num_batches)
            time.sleep(0.3)
            print('epoch {:2}/{:2} validation loss: {:5.5} '
                  'acc: {:5.5} '
                  'top5 acc: {:5.5} '
                  'acc_select: {:5.5},  '
                  'top 5 acc_select: {:5.5}'.
                  format(e, num_epochs, val_loss, accuracy, top5_accuracy, accuracy_select, top5_select_accuracy))
            print('saving checkpoint to: ', save_path)
            time.sleep(0.3)
            self._saver.save(sess, save_path, global_step=e * num_batches)

            p_bar = tqdm.tqdm(range(num_batches))
            for b in p_bar:
                is_validate = False
                feed_dict, _, _ = self._get_feed_dict(b, dataset, is_validate)
                fetches = {'train_op': self._train_op,
                           'loss': self._loss,
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
        detect_probs = self._detect_probs.eval(session=self._sess, feed_dict=feed_dict)
        return np.argmax(detect_probs, 1), detect_probs

    def selector_attack(self,seq, chosen_char_to_flip):
        flip_char = np.zeros_like(seq)
        flip_char[chosen_char_to_flip] = 1
        if len(seq.shape) == 1:
            seq = np.expand_dims(seq, 0)
            flip_char = np.expand_dims(flip_char, 0)
            flip_char = np.repeat(flip_char, 128, axis=0)
        feed_dict = {self._seq_ph: seq, self._lbl_ph: flip_char}
        select_probs = self._select_probs.eval(session=self._sess, feed_dict=feed_dict)
        return np.argmax(select_probs, 1)[0], select_probs[0]

def example():
    dataset = HotFlipDataProcessor.get_detector_selector_datasets()
    _, char_idx, _ = data.Dataset.init_embedding_from_dump()
    sess = tf.Session()
    config = FlipDetectorConfig(
        restore=False,
        restore_path=path.join(RES_OUT_DIR, 'detector_flip_beam_10/detector_model.ckpt-84056'))
    model = FlipDetector(sess, config=config)
    #model._validate(dataset)
    model.train(dataset)

    seq = dataset.train_seq[0]
    flip_idx, _ = model.attack(seq, target_confidence=0.)[0]

    sent = data.seq_2_sent(seq, char_idx)
    flipped_sent = sent[:flip_idx] + '[*]' + sent[min(flip_idx + 1, len(sent)):]
    print(sent)
    print(flipped_sent)
    sess.close()


if __name__ == '__main__':
    example()
