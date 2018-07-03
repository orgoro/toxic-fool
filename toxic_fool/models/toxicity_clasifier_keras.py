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
        return [(input_shape[0], output_len), (input_shape[0], input_shape[1])] # [atten_weighted_sum, atten_weights]


    # def compute_mask(self, input, input_mask=None):
    #     if isinstance(input_mask, list):
    #         return [None] * len(input_mask)
    #     else:
    #         return None


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


class CustomLoss(object):
    @staticmethod
    def binary_crossentropy_with_bias(recall_weight):
        def loss_function(y_true, y_pred):
            return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + recall_weight * K.sum(y_true * (1 - y_pred))

        return loss_function


class ToxicityClassifierKeras(ToxicityClassifier):
    # pylint: disable = too-many-arguments
    def __init__(self, session, max_seq, padded, num_tokens, embed_dim, embedding_matrix, recall_weight, metrics):
        # type: (tf.Session, np.int, bool) -> None
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._input_layer = None
        self._output_layer = None
        self._embedding = embedding_matrix
        self._recall_weight = recall_weight
        self._metrics = metrics
        super(ToxicityClassifierKeras, self).__init__(session=session, max_seq=max_seq, padded=padded)

    def embedding_layer(self, tensor):
        # TODO consider change to trainable=False
        emb = layers.Embedding(input_dim=self._num_tokens, output_dim=self._embed_dim, input_length=self._max_seq,
                               trainable=True, mask_zero=False, weights=[self._embedding])
        return emb(tensor)

    def spatial_dropout_layer(self, tensor, rate=0.25):
        dropout = layers.SpatialDropout1D(rate=rate)
        return dropout(tensor)

    def dropout_layer(self, tensor, rate=0.5):
        dropout = layers.Dropout(rate=rate)
        return dropout(tensor)

    def bidirectional_rnn(self, tensor, amount=60):
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
        atten = AttentionWeightedAverage()
        result = atten(tensor, mask=K.ones(1000))
        return result[0]

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
        atten = self.attention_layer(concat)
        all_views = self.concat_layer([last_stage, maxpool, avgpool, atten], axis=1)

        # classify:
        dropout2 = self.dropout_layer(all_views)
        dense = self.dense_layer(dropout2)
        self._output_layer = self.output_layer(dense)

        model = keras.Model(inputs=self._input_layer, outputs=self._output_layer)
        adam_optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
        model.compile(loss=CustomLoss.binary_crossentropy_with_bias(self._recall_weight), optimizer=adam_optimizer,
                      metrics=self._metrics)
        model.summary()
        return model

    def train(self, dataset):
        # type: (data.Dataset) -> None
        result = self._model.fit(x=dataset.train_seq[:, :], y=dataset.train_lbl[:, :], batch_size=500,
                                 validation_data=(dataset.val_seq, dataset.val_lbl))
        print(result)
        return result

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

        return fn([seq])


def example():
    sess = tf.Session()
    embedding_matrix = data.Dataset.init_embedding_from_dump()
    num_tokens, embed_dim = embedding_matrix.shape
    max_seq = 1000
    tox_model = ToxicityClassifierKeras(session=sess, max_seq=max_seq, num_tokens=num_tokens, embed_dim=embed_dim,
                                        padded=True, embedding_matrix=embedding_matrix, recall_weight=0.01,
                                        metrics=['accuracy', 'ce', CalcAccuracy.precision, CalcAccuracy.recall,
                                                 CalcAccuracy.f1])

    dataset = data.Dataset.init_from_dump()
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    grad_tox = tox_model.get_gradient(seq)[0]
    grad_norm = np.linalg.norm(grad_tox, axis=2)
    print('max grad location {}/{}'.format(np.argmax(grad_norm, axis=1), max_seq))

    classes = tox_model.classify(seq)
    print(classes)

    tox_model.train(dataset)
    with sess.as_default():
        (dataset.train_seq[0, :]).eval()
    classes = tox_model.classify(seq)
    seq = dataset.train_seq[0, :]
    grad_tox = tox_model.get_gradient(seq)[0]
    grad_norm = np.linalg.norm(grad_tox, axis=2)
    print('max grad location {}/{}'.format(np.argmax(grad_norm, axis=1), max_seq))

    print(classes)
    true_classes = dataset.train_lbl[0, :]
    print(true_classes)


if __name__ == '__main__':
    example()
