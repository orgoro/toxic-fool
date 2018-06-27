from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

EMBEDDING_DIR = path.dirname(path.abspath(__file__))
CHAR_EMBEDDING_PATH = path.join(EMBEDDING_DIR, 'glove.840B.300d-char.txt')
EMBEDDING_DIM = 300