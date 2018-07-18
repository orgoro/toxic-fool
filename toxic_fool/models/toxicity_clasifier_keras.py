from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras.engine import InputSpec, Layer
from keras import initializers
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import os
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

import data
from models.toxicity_clasifier import ToxicityClassifier


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.atten_weights = self.add_weight(shape=(input_shape[2], 1),
                                             name='{}_atten_weights'.format(self.name),
                                             initializer=self.init)
        self.trainable_weights = [self.atten_weights]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        mask = None
        for key, value in kwargs.items():
            if key == "mask":
                mask = value

        logits = K.dot(inputs, self.atten_weights)
        x_shape = K.shape(inputs)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = inputs * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        return [result, att_weights]

    def get_output_shape(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]  # [atten_weighted_sum, atten_weights]


class CalcAccuracy(object):
    @staticmethod
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(y_true)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def f1(y_true, y_pred):
        precision = CalcAccuracy.precision(y_true, y_pred)
        recall = CalcAccuracy.recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class RocCallback(Callback):
    def __init__(self, dataset):
        # type: (data.Dataset) -> None
        self.x = dataset.train_seq
        self.y = dataset.train_lbl
        self.x_val = dataset.val_seq
        self.y_val = dataset.val_lbl
        super(RocCallback, self).__init__()

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x[:1000])
        roc = roc_auc_score(self.y[:1000], y_pred[:1000], average='weighted')
        y_pred_val = self.model.predict(self.x_val[:1000])
        roc_val = roc_auc_score(self.y_val[:1000], y_pred_val[:1000], average='weighted')
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc, 4)), str(round(roc_val, 4))), end=100 * ' ' + '\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class CustomLoss(object):
    @staticmethod
    def binary_crossentropy_with_bias(recall_weight):
        def loss_function(y_true, y_pred):
            return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + recall_weight * K.sum(y_true * (1 - y_pred))

        return loss_function


class ToxicityClassifierKeras(ToxicityClassifier):
    # pylint: disable = too-many-arguments
    def __init__(self, session, max_seq, padded, num_tokens, embed_dim, embedding_matrix, metrics, args):
        # type: (tf.Session, np.int, bool, np.int, np.int, np.ndarray,list,np.typeDict) -> None
        if args is not None:
            self._restore_checkpoint = args.restore_checkpoint
            self._restore_checkpoint_fullpath = args.restore_checkpoint_fullpath
            self._save_checkpoint = args.save_checkpoint
            self._save_checkpoint_path = args.save_checkpoint_path
            self._use_gpu = args.use_gpu
            self._recall_weight = args.recall_weight
        else:
            self._recall_weight = 0.001
            self._use_gpu = False
            self._restore_checkpoint = False
            self._save_checkpoint = False
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._input_layer = None
        self._output_layer = None
        self._atten_w = None
        self._embedding = embedding_matrix
        self._metrics = metrics
        super(ToxicityClassifierKeras, self).__init__(session=session, max_seq=max_seq, padded=padded)

    def embedding_layer(self, tensor):
        # TODO consider change to trainable=False
        emb = layers.Embedding(input_dim=self._num_tokens, output_dim=self._embed_dim, input_length=self._max_seq,
                               trainable=False, mask_zero=False, weights=[self._embedding])
        return emb(tensor)

    def spatial_dropout_layer(self, tensor, rate=0.25):
        dropout = layers.SpatialDropout1D(rate=rate)
        return dropout(tensor)

    def dropout_layer(self, tensor, rate=0.7):
        dropout = layers.Dropout(rate=rate)
        return dropout(tensor)

    def bidirectional_rnn(self, tensor, amount=60):
        if self._use_gpu:
            bi_rnn = layers.Bidirectional(layers.CuDNNGRU(amount, return_sequences=True))
        else:
            bi_rnn = layers.Bidirectional(layers.GRU(amount, return_sequences=True))
        return bi_rnn(tensor)

    def concat_layer(self, tensors, axis):
        return layers.concatenate(tensors, axis=axis)

    def last_stage(self, tensor):
        last = layers.Lambda(lambda t: t[:, -1], name='last')
        return last(tensor)

    def max_polling_layer(self, tensor):
        maxpool = layers.GlobalMaxPooling1D()
        return maxpool(tensor)

    def avg_polling_layer(self, tensor):
        avgpool = layers.GlobalAveragePooling1D()
        return avgpool(tensor)

    def attention_layer(self, tensor):
        attenion = AttentionWeightedAverage()
        atten, atten_w = attenion(tensor)
        return atten, atten_w

    def dense_layer(self, tensor, out_size=144):
        dense = layers.Dense(out_size, activation='relu')
        return dense(tensor)

    def output_layer(self, tensor, out_size=6):
        output = layers.Dense(out_size, activation='sigmoid')
        return output(tensor)

    def _build_graph(self):
        K.set_session(self._session)

        # embed:
        self._input_layer = keras.Input(shape=(self._max_seq,), dtype='int32')
        self._embedding = self.embedding_layer(self._input_layer)
        dropout1 = self.spatial_dropout_layer(self._embedding)

        # rnn:
        rnn1 = self.bidirectional_rnn(dropout1)
        rnn2 = self.bidirectional_rnn(rnn1)
        concat = self.concat_layer([rnn1, rnn2], axis=2)

        # attentions:
        avgpool = self.avg_polling_layer(concat)
        maxpool = self.max_polling_layer(concat)
        last_stage = self.last_stage(concat)
        atten, self._atten_w = self.attention_layer(concat)
        all_views = self.concat_layer([last_stage, maxpool, avgpool, atten], axis=1)

        # classify:
        dropout2 = self.dropout_layer(all_views)
        dense = self.dense_layer(dropout2)
        self._output_layer = self.output_layer(dense)

        model = keras.Model(inputs=self._input_layer, outputs=self._output_layer)
        adam_optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
        if self._restore_checkpoint:
            if os.path.exists(self._restore_checkpoint_fullpath):
                model.load_weights(self._restore_checkpoint_fullpath)
                print("Restoring weights from " + self._restore_checkpoint_fullpath)
            else:
                print("Saved model was not fount at " + self._restore_checkpoint_fullpath + ", starting from scratch")
        model.compile(loss=CustomLoss.binary_crossentropy_with_bias(self._recall_weight), optimizer=adam_optimizer,
                      metrics=self._metrics)
        model.summary()
        return model

    def train(self, dataset):
        # type: (data.Dataset) -> keras.callbacks.History
        callback_list = []
        if self._save_checkpoint:
            if not os.path.isdir(self._save_checkpoint_path):
                os.mkdir(self._save_checkpoint_path)
            filepath = self._save_checkpoint_path + "/weights-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                         mode='max')
            callback_list.append(checkpoint)
        callback_list.append(RocCallback(dataset))
        history = self._model.fit(x=dataset.train_seq[:1000, :], y=dataset.train_lbl[:1000, :], batch_size=500,
                                  validation_data=(dataset.val_seq[:, :], dataset.val_lbl[:, :]), epochs=30,
                                  callbacks=callback_list)
        return history

    def classify(self, seq):
        # type: (np.ndarray) -> np.ndarray
        prediction = self._model.predict(seq)
        return prediction

    def get_f1_score(self, seqs, lbls):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        raise NotImplementedError('implemented by child')

    def get_gradient(self, seq):
        grad_0 = K.gradients(loss=self._model.output[:, 0], variables=self._embedding)[0]
        grad_1 = K.gradients(loss=self._model.output[:, 1], variables=self._embedding)[0]
        grad_2 = K.gradients(loss=self._model.output[:, 2], variables=self._embedding)[0]
        grad_3 = K.gradients(loss=self._model.output[:, 3], variables=self._embedding)[0]
        grad_4 = K.gradients(loss=self._model.output[:, 4], variables=self._embedding)[0]
        grad_5 = K.gradients(loss=self._model.output[:, 5], variables=self._embedding)[0]

        grads = [grad_0, grad_1, grad_2, grad_3, grad_4, grad_5]
        fn = K.function(inputs=[self._model.input], outputs=grads)

        return fn([seq])[0]

    def get_attention(self, seq):
        fn = K.function(inputs=[self._model.input], outputs=[self._atten_w])
        return fn([seq])[0]


def _visualize(history):
    # type: (keras.callbacks.History) -> None
    # Get training and test loss histories
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def _visualise_attention(seq, attention):
    first_char = np.nonzero(seq)[0][0]
    only_seq = seq[first_char:]
    input_length = len(only_seq)
    fig = plt.figure(figsize=(input_length / 5, 5))
    ax = fig.add_subplot(1, 1, 1)

    width = 20
    atten_map = np.tile(np.expand_dims(attention[first_char:], 0), reps=[width, 1])
    atten_map = np.repeat(atten_map, width, axis=1)
    ax.imshow(atten_map, cmap='plasma', interpolation='nearest'), plt.title('attention')
    x = list(np.arange(width / 2, width * (input_length + 0.5), width))
    ax.set_xticks(x)
    ax.set_xticklabels(only_seq, rotation=45, fontdict={'fontsize': 8})
    plt.show()


def example(args):
    sess = tf.Session()
    embedding_matrix = data.Dataset.init_embedding_from_dump()
    num_tokens, embed_dim = embedding_matrix.shape
    max_seq = 400

    tox_model = ToxicityClassifierKeras(session=sess, max_seq=max_seq, num_tokens=num_tokens, embed_dim=embed_dim,
                                        padded=True, embedding_matrix=embedding_matrix,
                                        metrics=['accuracy', 'ce', CalcAccuracy.precision, CalcAccuracy.recall,
                                                 CalcAccuracy.f1], args=args)

    dataset = data.Dataset.init_from_dump()
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    grad_tox = tox_model.get_gradient(seq)
    grad_norm = np.linalg.norm(grad_tox, axis=2)
    print('max grad location {}/{}'.format(np.argmax(grad_norm, axis=1), max_seq))

    classes = tox_model.classify(seq)
    atten_w = tox_model.get_attention(seq)
    _visualise_attention(seq[0], atten_w[0])
    print(classes)

    history = tox_model.train(dataset)
    grad_tox = tox_model.get_gradient(seq)
    grad_norm = np.linalg.norm(grad_tox, axis=2)
    print('max grad location {}/{}'.format(np.argmax(grad_norm, axis=1), max_seq))

    classes = tox_model.classify(seq)
    atten_w = tox_model.get_attention(seq)
    _visualise_attention(seq[0], atten_w[0])
    print(classes)
    true_classes = dataset.train_lbl[0, :]
    print(true_classes)

    _visualize(history=history)


if __name__ == '__main__':
    example(args=None)
