from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import fire
from os import path
from collections import defaultdict
import numpy as np

from resources_out import RES_OUT_DIR
import resources as res


def analyze_forms(form_paths, output_path=RES_OUT_DIR):
    assert isinstance(form_paths, list), 'forms paths should be a list'
    for form_path in form_paths:
        assert path.exists(form_path), 'form does not exist: {}'.format(form_path)
    assert isinstance(form_paths, list), 'forms paths should be a list'

    results_read = defaultdict(list)
    results_tox = defaultdict(list)
    for form_path in form_paths:
        csv_data = pd.read_csv(form_path)
        col_iter = iter(csv_data.columns)
        _ = next(col_iter)
        _ = next(col_iter)
        q_num = 0
        while True:
            try:
                sent_tox = next(col_iter)
                sent_readability = next(col_iter)
                q_idx = sent_tox.split(')')[0]
                q_class = _get_q_class(q_idx)
                toxicity = 0
                try:
                    toxicity = 5 * csv_data[sent_tox].value_counts()['Toxic'] / len(csv_data[sent_tox])

                except KeyError:
                    pass
                readability = csv_data[sent_readability].mean()
                print(f'{q_idx:5} | class {q_class:7} \ttoxicity: {toxicity:1} \t readability: {readability:1.3}')
                results_read[q_class] += [readability]
                results_tox[q_class] += [toxicity]
                q_num += 1
            except StopIteration:
                print('form: {} done after {} questions aggregated results so far'.format(form_path, q_num))
                break

    _print_summary(results_read, results_tox)


def _print_summary(results_read, results_tox):
    print('SUMMARY:')
    print('-' * 65)
    print(f'question_class\t\t\t |\treadable results |\tcount')
    print('-' * 65)
    for q_class, val_list in results_read.items():
        avg_readability = np.mean(np.asarray(val_list)/5)
        std_readability = np.std(np.asarray(val_list)/5)
        count = len(val_list)
        print(f'{q_class}|\t{avg_readability:01.3f} (+/- {std_readability:1.3f})|\t{count}')
    print('-' * 65)
    print('-' * 65)
    print(f'question_class\t\t\t |\ttoxicity results |\tcount')
    print('-' * 65)
    for q_class, val_list in results_tox.items():
        avg_toxicity = np.mean(np.asarray(val_list)/5)
        std_toxicity = np.std(np.asarray(val_list)/5)
        count = len(val_list)
        print(f'{q_class:25}|\t{avg_toxicity:1.3f} (+/- {std_toxicity:1.3f})|\t{count}')


def _get_q_class(q_idx):
    if q_idx.endswith('1'):
        q_class = '{:25}'.format('Non-Toxic')
    elif q_idx.endswith('9'):
        q_class = '{:25}'.format('Toxic')
    elif q_idx.endswith('4'):
        q_class = '{:25}'.format('Attacker')
    elif q_idx.endswith('7'):
        q_class = '{:25}'.format('Attacker-Duplication')
    else:
        raise ValueError('unknown class from index: {}'.format(q_idx))
    return q_class


if __name__ == '__main__':
    analyze_forms([res.FORMS_COL3, res.FORMS_COL2, res.FORMS_COL1])
    # fire.Fire(analyze_forms)
