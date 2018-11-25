from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

from .embedding import  CHAR_EMBEDDING_PATH, EMBEDDING_DIR , EMBEDDING_DIM
RESOURCES_DIR = path.dirname(path.abspath(__file__))

TEST_LBLS_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'test_labels.csv')
TEST_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'test.csv')
TRAIN_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'train.csv')

LATEST_KERAS_WEIGHTS = path.join(RESOURCES_DIR, 'data', 'weights.latest.hdf5')
LATEST_DETECTOR_WEIGHTS = path.join(RESOURCES_DIR, 'data', 'detector_flip_beam_10/detector_model.ckpt-84056')

