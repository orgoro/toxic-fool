from __future__ import absolute_import
import argparse
from models.toxicity_clasifier_keras import example
from resources_out import RES_OUT_DIR
from os import path


if __name__ == '__main__':
    # Parse cmd line arguments
    # pylint: disable=C0103
    parser = argparse.ArgumentParser()
    parser.add_argument('-restore_checkpoint', action='store_true', default=False, dest='restore_checkpoint',
                        help='Whether to restore previously saved checkpoint')
    parser.add_argument('-restore_checkpoint_fullpath', action="store", default= path.join(RES_OUT_DIR,'weights.latest.hdf5'),
                        dest="restore_checkpoint_fullpath", help='Full path of the checkpoint file to restore')
    parser.add_argument('-save_checkpoint', action='store_true', default=False, dest='save_checkpoint',
                        help='Whether to save checkpoints at the end of each epoch')
    parser.add_argument('-save_checkpoint_path', action="store", default=RES_OUT_DIR,
                        dest="save_checkpoint_path", help='Path of the checkpoint directory to save')
    parser.add_argument('-use_gpu', action='store_true', default=False, dest='use_gpu',
                        help='Whether to use gpu')
    parser.add_argument('-recall_weight', action='store',type=float, default=0.001, dest='recall_weight',
                        help='Recall weight in loss function')
    args = parser.parse_args()

    example(args=args)

