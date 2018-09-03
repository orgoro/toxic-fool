from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re

ROW_0 = '`1234567890-='
ROW_1 = 'qwertyuiop[]\\'
ROW_2 = 'asdfghjkl;\''
ROW_3 = 'zxcvbnm,./'

ROW_0_U = '~!@#$%^&*()_+'
ROW_1_U = 'QWERTYUIOP{}|'
ROW_2_U = 'ASDFGHJKL:"'
ROW_3_U = 'ZXCVBNM<>?'

ROWS = [ROW_0, ROW_1, ROW_2, ROW_3]
ROW_U = [ROW_0_U, ROW_1_U, ROW_2_U, ROW_3_U]


def _get_char_neighbors(is_upper, row_idx, char_idx):
    if is_upper:
        cur_rows = ROW_U
    else:
        cur_rows = ROWS

    min_row = max(0, row_idx - 1)
    max_row = min(3, row_idx + 1)

    pos_chars = ''
    for r in range(min_row, max_row + 1):
        row_len = len(cur_rows[r])
        if 0 < char_idx < row_len:
            pos_chars += cur_rows[r][char_idx - 1]
        if r != row_idx:
            if char_idx < row_len - 1:
                pos_chars += cur_rows[r][char_idx]
        if char_idx < row_len - 2:
            pos_chars += cur_rows[r][char_idx + 1]

    return pos_chars


def _find_char(char,):
    in_row_0 = ROW_0.find(char)
    in_row_1 = ROW_1.find(char)
    in_row_2 = ROW_2.find(char)
    in_row_3 = ROW_3.find(char)

    in_row_0_u = ROW_0_U.find(char)
    in_row_1_u = ROW_1_U.find(char)
    in_row_2_u = ROW_2_U.find(char)
    in_row_3_u = ROW_3_U.find(char)

    found = False
    is_upper = False
    row_idx = -1
    char_idx = -1

    if in_row_0 != -1:
        found, is_upper, row_idx, char_idx = (True, False, 0, in_row_0)
    if in_row_1 != -1:
        found, is_upper, row_idx, char_idx = (True, False, 1, in_row_1)
    if in_row_2 != -1:
        found, is_upper, row_idx, char_idx = (True, False, 2, in_row_2)
    if in_row_3 != -1:
        found, is_upper, row_idx, char_idx = (True, False, 3, in_row_3)

    if in_row_0_u != -1:
        found, is_upper, row_idx, char_idx = (True, True, 0, in_row_0_u)
    if in_row_1_u != -1:
        found, is_upper, row_idx, char_idx = (True, True, 1, in_row_1_u)
    if in_row_2_u != -1:
        found, is_upper, row_idx, char_idx = (True, True, 2, in_row_2_u)
    if in_row_3_u != -1:
        found, is_upper, row_idx, char_idx = (True, True, 3, in_row_3_u)

    if not found:
        raise ValueError('char not found: ', char)

    return is_upper, row_idx, char_idx

def get_possible_replace(char,preserve_type=True):
    is_upper, row_idx, char_idx = _find_char(char)
    neighbours = _get_char_neighbors(is_upper, row_idx, char_idx)
    if preserve_type:
        if char.isalpha():
            neighbours = re.sub(r'[^a-zA-Z]', '', neighbours)
        else:
            neighbours = re.sub(r'[a-zA-Z]', '', neighbours)
    return neighbours

def smart_replace(char, preserve_type=True):
    #type: (str, bool) -> str
    neighbours = get_possible_replace(char,preserve_type)
    np.random.seed(42)
    selected_char = np.random.randint(0, len(neighbours))
    return neighbours[selected_char]


def example():
    sent = 'Hey this is what I think!'
    print(sent)
    sent = smart_replace(sent[0]) + sent[1:6] + smart_replace(sent[6]) + sent[7:]
    print(sent)


if __name__ == '__main__':
    example()
