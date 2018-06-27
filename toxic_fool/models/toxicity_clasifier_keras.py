from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers

import data
from models.toxicity_clasifier import ToxicityClassifier


class ToxicityClassifierKeras(ToxicityClassifier):

    def __init__(self, session, max_seq, padded, num_tokens, embed_dim,embedding_matrix):
        # type: (tf.Session, np.int, bool) -> None
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._input_layer = None
        self._output_layer = None
        self._embedding = embedding_matrix
        super(ToxicityClassifierKeras, self).__init__(session=session, max_seq=max_seq, padded=padded)

    def embedding_layer(self, tensor):
        # TODO consider change to trainable=False
        emb = layers.Embedding(input_dim=self._num_tokens, output_dim=self._embed_dim, input_length=self._max_seq,
                               trainable=True, mask_zero=False , weights=[self._embedding])
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
        raise NotImplementedError

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
        # TODO: add attention atten = self.attention_layer(concat)
        all_views = self.concat_layer([last_stage, maxpool, avgpool], axis=1)

        # classify:
        dropout2 = self.dropout_layer(all_views)
        dense = self.dense_layer(dropout2)
        self._output_layer = self.output_layer(dense)

        model = keras.Model(inputs=self._input_layer, outputs=self._output_layer)
        adam_optimizer = keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
        model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy', 'ce'])
        model.summary()
        return model

    def train(self, dataset):
        # type: (data.Dataset) -> None
        result = self._model.fit(x=dataset.train_seq[:3001, :], y=dataset.train_lbl[:3001, :], batch_size=500,
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
    num_tokens , embed_dim = embedding_matrix.shape
    max_seq = 1000
    tox_model = ToxicityClassifierKeras(session=sess, max_seq=max_seq, num_tokens=num_tokens, embed_dim=embed_dim,
                                        padded=True,embedding_matrix = embedding_matrix)


    dataset = data.Dataset.init_from_dump()
    seq = np.expand_dims(dataset.train_seq[0, :], 0)
    grad_tox = tox_model.get_gradient(seq)[0]
    grad_norm = np.linalg.norm(grad_tox, axis=2)
    print('max grad location {}/{}'.format(np.argmax(grad_norm, axis=1), max_seq))

    classes = tox_model.classify(seq)
    print(classes)

    tox_model.train(dataset)
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
