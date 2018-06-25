from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

RESOURCES_DIR = path.dirname(path.abspath(__file__))

TEST_LBLS_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'test_labels.csv')
TEST_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'test.csv')
TRAIN_CSV_PATH = path.join(RESOURCES_DIR, 'data', 'train.csv')
