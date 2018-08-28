from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from attacks.hot_flip import HotFlip  ##needed to load hot flip data
from agents.flip_detector import FlipDetector, FlipDetectorConfig
from agents.smart_replace import smart_replace , get_possible_replace
from models import ToxicityClassifierKeras
from models import ToxClassifierKerasConfig
from agents.agent import AgentConfig
import random
import data
import tensorflow as tf
import numpy as np
from resources_out import RES_OUT_DIR
from os import path

SPACE_EMBEDDING = 95
MAX_SEQ = 400


def create_token_dict(char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    return token_index


class RandomFlip(object):
    def attack(self, seq, mask, token_index, char_index,make_smart_replace=True):
        assert len(mask) == len(seq)
        masked_seq = seq * mask
        spaces_indices = np.where(seq == SPACE_EMBEDDING)
        masked_seq[spaces_indices] = 0
        if not masked_seq.any():
            return 0,0,0,seq
        char_idx_to_flip = random.choice(np.where(masked_seq != 0)[0])
        char_token_to_flip = seq[char_idx_to_flip]
        char_to_flip = token_index[char_token_to_flip]
        if make_smart_replace:
            char_to_flip_to = smart_replace(char_to_flip)
            token_to_flip_to = char_index[char_to_flip_to]
        else:
            token_to_flip_to = char_token_to_flip
            while token_to_flip_to == char_token_to_flip:
                token_to_flip_to = np.random.randint(1,SPACE_EMBEDDING)
            char_to_flip_to = token_index[token_to_flip_to]
        flipped_seq = seq
        original_char = seq[char_idx_to_flip]
        flipped_seq[char_idx_to_flip] = token_to_flip_to
        return original_char, char_to_flip_to, char_idx_to_flip, flipped_seq


class AttackerConfig(AgentConfig):
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_of_sen_to_attack=100,
                 attack_until_break=True,
                 debug=True,
                 smart_replace=True,
                 flip_once_in_a_word=False,
                 flip_middle_letters_only=False):
        self.num_of_sen_to_attack = num_of_sen_to_attack
        self.attack_until_break = attack_until_break
        self.debug = debug
        self.smart_replace = smart_replace
        self.flip_once_in_a_word = flip_once_in_a_word
        self.flip_middle_letters_only = flip_middle_letters_only

        super(AttackerConfig, self).__init__()


class Attacker(object):

    def __init__(self,
                 session,
                 tox_model=None,
                 hotflip=None,
                 flip_detector=None,
                 random_flip=None,
                 config=AttackerConfig()):
        self._sess = session
        if flip_detector:
            self._flip_detector = flip_detector
        else:
            flip_detector_config = FlipDetectorConfig(eval_only=True)
            self._flip_detector = FlipDetector(self._sess, config=flip_detector_config)
        if tox_model:
            self._tox_model = tox_model
        else:
            tox_config = ToxClassifierKerasConfig(debug=False)
            self._tox_model = ToxicityClassifierKeras(self._sess, config=tox_config)
            self._hotflip = hotflip if hotflip else HotFlip(model=self._tox_model,
                                                            num_of_char_to_flip = 1,
                                                            beam_search_size = 1,
                                                            only_smart_replace_allowed = config.smart_replace,
                                                            debug= False)

        self._random_flip = random_flip if random_flip else RandomFlip()
        self.config = config
        self.dataset = data.Dataset.init_from_dump()
        _, char_index, _ = data.Dataset.init_embedding_from_dump()
        self.char_index = char_index
        self.token_index = create_token_dict(char_index)

    # pylint: disable=dangerous-default-value
    def attack(self, model='random', seq=None, mask=[], sequence_idx=0):
        assert model in ['random', 'hotflip', 'detector']
        assert seq is not None
        seq = seq.copy()
        curr_seq = seq[sequence_idx]
        if len(mask) == 0:
            mask = np.ones_like(curr_seq)
        sent = data.seq_2_sent(curr_seq, self.char_index)
        tox_before = self._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0]
        if self.config.debug:
            print("Attacking with model: ", model)
            print("Toxicity before attack: ", tox_before)
            print(sent)
        if model == 'random':
            _, _, flip_idx, res = self._random_flip.attack(curr_seq,
                                                           mask,
                                                           self.token_index,
                                                           self.char_index,
                                                           make_smart_replace=self.config.smart_replace)
        elif model == 'hotflip':

            res =    self._hotflip.attack(np.expand_dims(curr_seq, 0),mask)
            flip_idx = res[0].char_to_flip_to
            res = res[0].fliped_sent
        else:
            _, probs = self._flip_detector.attack(curr_seq, target_confidence=0.)
            spaces_indices = np.where(curr_seq == SPACE_EMBEDDING)
            mask[spaces_indices] = 0
            mask_probs = probs * mask
            flip_idx = np.argmax(mask_probs, 1)[0]
            token_to_flip = curr_seq[flip_idx]
            if not mask.any():
                return 0,0,0,curr_seq
            char_to_flip = self.token_index[token_to_flip]
            if self.config.smart_replace:
                char_to_flip_to = smart_replace(char_to_flip)
                token_of_flip = self.char_index[char_to_flip_to]
            else:
                token_of_flip = token_to_flip
                while token_of_flip == token_to_flip:
                    token_of_flip = np.random.randint(1,SPACE_EMBEDDING)
            curr_seq[flip_idx] = token_of_flip
            res = curr_seq
        flipped_sent = data.seq_2_sent(res, self.char_index)
        tox_after = self._tox_model.classify(np.expand_dims(res, 0))[0][0]
        if self.config.debug:
            print("Toxicity after attack: ", tox_after)
            print(flipped_sent)
        return tox_before, tox_after, flip_idx, res

    def remove_word_from_mask(self, flipped_seq, mask, flip_idx):
        seq_start = flipped_seq[:flip_idx]
        seq_end   = flipped_seq[flip_idx+1:]
        reversed_seq_start = seq_start[::-1]
        if SPACE_EMBEDDING in seq_end:
            space_fw_offset = np.where(seq_end == SPACE_EMBEDDING)[0][0]
        else:
            space_fw_offset = len(seq_end)
        if SPACE_EMBEDDING in reversed_seq_start:
            space_bw_offset = np.where(reversed_seq_start == SPACE_EMBEDDING)[0][0]
        else:
            space_bw_offset = len(np.where(reversed_seq_start != 0))-1
        word_start = flip_idx - space_bw_offset - 1
        word_end =  flip_idx + space_fw_offset + 1
        mask[word_start:word_end] = 0
        return mask

    def remove_first_and_last_word_letters(self,flipped_seq,mask):
        space_indices = np.where(flipped_seq == SPACE_EMBEDDING)
        space_indices_plus = np.add(space_indices, 1)
        space_indices_minus = np.subtract(space_indices, 1)
        first_char = np.min(np.where(flipped_seq != 0))
        mask[space_indices_plus] = 0
        mask[space_indices_minus] = 0
        mask[first_char] = 0
        mask[MAX_SEQ-1] = 0
        return mask


    def attack_until_break(self,
                           model='random',
                           seq=None,
                           mask=None,
                           sequence_idx=0):
        seq = seq.copy()
        curr_seq = seq[sequence_idx]
        tox = self._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0]
        cnt = 0
        cant_untoxic = 0
        curr_seq_copy = curr_seq.copy()
        curr_seq_space_indices = np.where(curr_seq_copy == SPACE_EMBEDDING)
        curr_seq_copy[curr_seq_space_indices] = 0
        curr_seq_replacable_chars = np.sum(curr_seq_copy != 0)
        if not mask:
            non_letters = np.where(curr_seq == 0)
            mask = np.ones_like(curr_seq)
            mask[non_letters] = 0
        if self.config.flip_middle_letters_only:
            mask = self.remove_first_and_last_word_letters(curr_seq, mask)
        while tox > 0.5:
            cnt += 1
            _, tox, flip_idx, flipped_seq = self.attack(model=model,
                                                        seq=seq,
                                                        mask=mask,
                                                        sequence_idx=sequence_idx)
            if np.array_equal(seq[sequence_idx],flipped_seq) or cnt == curr_seq_replacable_chars - 1:
                print ("Replaced all chars and couldn't change sentence to non toxic with model ", model)
                cant_untoxic = 1
                break
            if self.config.flip_once_in_a_word:
                mask = self.remove_word_from_mask(flipped_seq, mask, flip_idx)
            if self.config.flip_middle_letters_only:
                mask = self.remove_first_and_last_word_letters(flipped_seq,mask)
            seq[sequence_idx] = flipped_seq
            mask[flip_idx] = 0
        if self.config.debug:
            print("Toxicity after break: ", tox)
            print("Number of flips needed: ", cnt)
        return tox, cnt, cant_untoxic


def example():
    dataset = data.Dataset.init_from_dump()
    sess = tf.Session()
    attacker_config = AttackerConfig(debug=True,
                                     flip_once_in_a_word=False,
                                     flip_middle_letters_only=True,
                                     smart_replace=False)
    attacker = Attacker(session=sess, config=attacker_config)

    index_of_toxic_sent = np.where(dataset.val_lbl[:, 0] == 1)[0]

    attack_list = []
    attack_list.append((dataset.val_seq[index_of_toxic_sent], dataset.val_lbl[index_of_toxic_sent], 'val'))

    seq, _, _ = attack_list[0]

    # attacker.attack(seq=seq, model='random', labels=label)
    # attacker.attack(seq=seq, model='hotflip', labels=label)
    # attacker.attack(seq=seq, model='detector', labels=label)
    random_cnt_list = list()
    hotflip_cnt_list = list()
    detector_cnt_list = list()
    detector_cant_untoxic_cnt = 0
    random_cant_untoxic_cnt = 0
    for j in range(100):
        print ("Working on sentence ", j)
        curr_seq = seq[j]
        if attacker._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0] < 0.5:
            continue
        _, random_cnt, random_cant_untoxic = attacker.attack_until_break(model='random',
                                                                         seq=seq,
                                                                         sequence_idx=j)
        random_cnt_list.append(random_cnt)
        _, hotflip_cnt,_ = attacker.attack_until_break(model=
                                                       'hotflip',
                                                       seq=seq,
                                                       sequence_idx=j)
        hotflip_cnt_list.append(hotflip_cnt)
        _, detector_cnt, detector_cant_untoxic = attacker.attack_until_break(model='detector',
                                                                             seq=seq,
                                                                             sequence_idx=j)
        detector_cnt_list.append(detector_cnt)
        detector_cant_untoxic_cnt += detector_cant_untoxic
        random_cant_untoxic_cnt += random_cant_untoxic
        print ("Random Cnt: ", random_cnt)
        print ("Hotflip Cnt: ", hotflip_cnt)
        print ("Detector Cnt: ", detector_cnt)

    print ("Random mean: ", np.mean(random_cnt_list))
    print("Num of sentences cant untoxic random: ", str(random_cant_untoxic_cnt))
    print ("Hotflip mean: ", np.mean(hotflip_cnt_list))
    print ("Detector mean: ", np.mean(detector_cnt_list))
    print("Num of sentences cant untoxic detector: ", str(detector_cant_untoxic_cnt))
    flips_cnt = dict()
    flips_cnt['random'] = random_cnt_list
    flips_cnt['hotflip'] = hotflip_cnt_list
    flips_cnt['detector'] = detector_cnt_list
    np.save(path.join(RES_OUT_DIR, 'flips_cnt.npy'), flips_cnt )

if __name__ == '__main__':
    example()
