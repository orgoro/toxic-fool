from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import path
from attacks.hot_flip import HotFlip  ##needed to load hot flip data
from agents.flip_detector import FlipDetector, FlipDetectorConfig
from agents.smart_replace import smart_replace, get_possible_replace
from models import ToxicityClassifierKeras
from models import ToxClassifierKerasConfig
from agents.agent import AgentConfig
import random
import data
import tensorflow as tf
import numpy as np
from resources_out import RES_OUT_DIR
import time

SPACE_EMBEDDING = 95
MAX_SEQ = 500


def create_token_dict(char_idx):
    # convert the char to token dic into token to char dic
    token_index = {}
    for key, value in char_idx.items():
        token_index[value] = key

    return token_index


class RandomFlip(object):
    def attack(self, seq, mask, token_index, char_index, make_smart_replace=True):
        assert len(mask) == len(seq)
        masked_seq = seq * mask
        spaces_indices = np.where(seq == SPACE_EMBEDDING)
        masked_seq[spaces_indices] = 0
        if not masked_seq.any():
            return 0, 0, 0, seq
        char_idx_to_flip = random.choice(np.where(masked_seq != 0)[0])
        char_token_to_flip = seq[char_idx_to_flip]
        char_to_flip = token_index[char_token_to_flip]
        if make_smart_replace:
            char_to_flip_to = smart_replace(char_to_flip)
            token_to_flip_to = char_index[char_to_flip_to]
        else:
            token_to_flip_to = char_token_to_flip
            while token_to_flip_to == char_token_to_flip:
                token_to_flip_to = np.random.randint(1, SPACE_EMBEDDING)
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
                                                            num_of_char_to_flip=1,
                                                            beam_search_size=1,
                                                            only_smart_replace_allowed=config.smart_replace,
                                                            debug=False)

        self._random_flip = random_flip if random_flip else RandomFlip()
        self.config = config
        self.dataset = data.Dataset.init_from_dump()
        _, char_index, _ = data.Dataset.init_embedding_from_dump()
        self.char_index = char_index
        self.token_index = create_token_dict(char_index)
        self.atten_fn = self._tox_model.get_attention_fn()
        self.attack_list = list()

    # pylint: disable=dangerous-default-value
    def attack(self, model='random', seq=None, mask=[], sequence_idx=0, attack_number=0):
        assert model in ['random', 'hotflip', 'detector', 'atten']
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
        time_before = time.time()
        if model == 'random':
            _, _, flip_idx, res = self._random_flip.attack(curr_seq,
                                                           mask,
                                                           self.token_index,
                                                           self.char_index,
                                                           make_smart_replace=self.config.smart_replace)
            time_for_attack = time.time() - time_before
        elif model == 'hotflip':
            res = self._hotflip.attack(np.expand_dims(curr_seq, 0), mask)
            time_for_attack = time.time() - time_before
            flip_idx = res[0].char_to_flip_to
            res = res[0].fliped_sent
        elif model == 'atten':
            # atten_probs = np.zeros_like(curr_seq)
            spaces_indices = np.where(curr_seq == SPACE_EMBEDDING)
            mask[spaces_indices] = 0
            atten_probs = self.atten_fn([np.expand_dims(curr_seq, 0)])[0]
            time_for_attack = time.time() - time_before
            atten_probs = np.squeeze(atten_probs)
            atten_probs_masked = atten_probs * mask
            flip_idx = np.argsort(atten_probs_masked)[-1]
            token_to_flip = curr_seq[flip_idx]
            char_to_flip = self.token_index[token_to_flip]
            if not atten_probs_masked.any():
                res = curr_seq
            else:
                char_to_flip_to = smart_replace(char_to_flip)
                token_of_flip = self.char_index[char_to_flip_to]
                curr_seq[flip_idx] = token_of_flip
                res = curr_seq
        else:
            _, probs = self._flip_detector.attack(curr_seq, target_confidence=0.)
            time_for_attack = time.time() - time_before
            spaces_indices = np.where(curr_seq == SPACE_EMBEDDING)
            mask[spaces_indices] = 0
            mask_probs = probs * mask
            flip_idx = np.argmax(mask_probs, 1)[0]
            token_to_flip = curr_seq[flip_idx]
            if not mask.any():
                return 0, 0, 0, curr_seq, time_for_attack
            char_to_flip = self.token_index[token_to_flip]
            if self.config.smart_replace:
                char_to_flip_to = smart_replace(char_to_flip)
                token_of_flip = self.char_index[char_to_flip_to]
            else:
                token_of_flip,_ = self._flip_detector.selector_attack(curr_seq, flip_idx)
                # token_of_flip = token_to_flip
                # while token_of_flip == token_to_flip:
                #     token_of_flip = np.random.randint(1, SPACE_EMBEDDING)
            curr_seq[flip_idx] = token_of_flip
            res = curr_seq
        flipped_sent = data.seq_2_sent(res, self.char_index)
        tox_after = self._tox_model.classify(np.expand_dims(res, 0))[0][0]
        if self.config.debug:
            print("Toxicity after attack: ", tox_after)
            print(flipped_sent)
        single_attack = dict()
        single_attack['seq_idx'] = sequence_idx
        single_attack['seq_length'] = np.count_nonzero(curr_seq)
        single_attack['attack_model'] = model
        single_attack['time_for_attack'] = time_for_attack
        single_attack['tox_before'] = tox_before
        single_attack['tox_after'] = tox_after
        single_attack['attack_number'] = attack_number
        single_attack['smart_replace'] = self.config.smart_replace
        single_attack['flip_once_in_a_word'] = self.config.flip_once_in_a_word
        single_attack['flip_middle_letters_only'] = self.config.flip_middle_letters_only
        self.attack_list.append(single_attack.copy())
        return tox_before, tox_after, flip_idx, res, time_for_attack

    def remove_word_from_mask(self, flipped_seq, mask, flip_idx):
        seq_start = flipped_seq[:flip_idx]
        seq_end = flipped_seq[flip_idx + 1:]
        reversed_seq_start = seq_start[::-1]
        if SPACE_EMBEDDING in seq_end:
            space_fw_offset = np.where(seq_end == SPACE_EMBEDDING)[0][0]
        else:
            space_fw_offset = len(seq_end)
        if SPACE_EMBEDDING in reversed_seq_start:
            space_bw_offset = np.where(reversed_seq_start == SPACE_EMBEDDING)[0][0]
        else:
            space_bw_offset = len(np.where(reversed_seq_start != 0)) - 1
        word_start = flip_idx - space_bw_offset - 1
        word_end = flip_idx + space_fw_offset + 1
        mask[word_start:word_end] = 0
        return mask

    def remove_first_and_last_word_letters(self, flipped_seq, mask):
        space_indices = np.where(flipped_seq == SPACE_EMBEDDING)[0]
        space_indices_plus = np.add(space_indices, 1)
        space_indices_minus = np.subtract(space_indices, 1)
        first_char = np.min(np.where(flipped_seq != 0))
        if MAX_SEQ - 1 in space_indices:
            last_space_index = np.where(space_indices_plus == MAX_SEQ)
            space_indices_plus = np.delete(space_indices_plus, last_space_index)
        mask[space_indices_plus] = 0
        mask[space_indices_minus] = 0
        mask[first_char] = 0
        mask[MAX_SEQ - 1] = 0
        return mask

    def attack_hot_flip_until_break(self,   seq, beam_size ,  sequence_idx=0):

        time_before = time.time()



        seq = seq.copy()
        curr_seq = seq[sequence_idx]
        hot_flip = HotFlip(model=self._tox_model , break_on_half=True,beam_search_size=beam_size)
        best_flip_status, _ = hot_flip.attack(seq=np.expand_dims(curr_seq, 0))


        sent_attacks = []
        #reverse list
        while best_flip_status.prev_flip_status != None: ##the original sentence has prev_flip_status = None
            sent_attacks.append(best_flip_status.fliped_sent)
            best_flip_status = best_flip_status.prev_flip_status

        num_of_flips = len(sent_attacks)


        time_for_attack = time.time() - time_before

        return num_of_flips , [time_for_attack / float(num_of_flips)] * num_of_flips

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
        time_for_attack_list = list()
        if not mask:
            non_letters = np.where(curr_seq == 0)
            mask = np.ones_like(curr_seq)
            mask[non_letters] = 0
        if self.config.flip_middle_letters_only:
            mask = self.remove_first_and_last_word_letters(curr_seq, mask)
        while tox > 0.5:
            cnt += 1
            _, tox, flip_idx, flipped_seq, time_for_attack = self.attack(model=model,
                                                                         seq=seq,
                                                                         mask=mask,
                                                                         sequence_idx=sequence_idx,
                                                                         attack_number=cnt)
            time_for_attack_list.append(time_for_attack)
            if np.array_equal(seq[sequence_idx], flipped_seq) or cnt == curr_seq_replacable_chars - 1:
                print("Replaced all chars and couldn't change sentence to non toxic with model ", model)
                cant_untoxic = 1
                break
            if self.config.flip_once_in_a_word:
                mask = self.remove_word_from_mask(flipped_seq, mask, flip_idx)
            if self.config.flip_middle_letters_only:
                mask = self.remove_first_and_last_word_letters(flipped_seq, mask)
            seq[sequence_idx] = flipped_seq
            mask[flip_idx] = 0
        if self.config.debug:
            print("Toxicity after break: ", tox)
            print("Number of flips needed: ", cnt)
        return cnt, cant_untoxic, time_for_attack_list


def example():
    dataset = data.Dataset.init_from_dump()
    sess = tf.Session()
    attacker_config = AttackerConfig(debug=False,
                                     flip_once_in_a_word=False,
                                     flip_middle_letters_only=False,
                                     smart_replace=False)
    attacker = Attacker(session=sess, config=attacker_config)

    index_of_toxic_sent = np.where(dataset.val_lbl[:, 0] == 1)[0]

    attack_list = []
    attack_list.append((dataset.val_seq[index_of_toxic_sent], dataset.val_lbl[index_of_toxic_sent], 'val'))

    seq, _, _ = attack_list[0]

    # attacker.attack(seq=seq, model='random', labels=label)
    # attacker.attack(seq=seq, model='hotflip', labels=label)
    # attacker.attack(seq=seq, model='detector', labels=label)
    random_cnt_list_moderate = list()
    hotflip_cnt_list_moderate = list()
    hotflip_beam5_cnt_list_moderate = list()
    hotflip_beam10_cnt_list_moderate = list()
    detector_cnt_list_moderate = list()
    atten_cnt_list_moderate = list()
    random_cnt_list_hard = list()
    hotflip_cnt_list_hard = list()
    detector_cnt_list_hard = list()
    atten_cnt_list_hard = list()
    hotflip_cnt_list_no_smart = list()
    detector_cant_untoxic_cnt = 0
    random_cant_untoxic_cnt = 0
    atten_cant_untoxic_cnt = 0
    # random_time_for_attack_list = list()
    # atten_time_for_attack_list = list()
    hotflip_beam5_time_for_attack_list = list()
    hotflip_beam10_time_for_attack_list = list()
    detector_time_for_attack_list = list()
    hotflip_time_for_attack_list = list()
    sentences_to_run = range(400)
    for j in range(len(sentences_to_run)):
        print("Working on sentence ", j)
        curr_seq = seq[sentences_to_run[j]]
        if attacker._tox_model.classify(np.expand_dims(curr_seq, 0))[0][0] < 0.5:
            continue
        ## Attack all models with moderate attack
        attacker.config.smart_replace = False
        attacker.config.flip_middle_letters_only = False
        attacker.config.flip_once_in_a_word = False

        # random_cnt_moderate, random_cant_untoxic, random_time_for_attack = attacker.attack_until_break(model='random',
        #                                                                                                seq=seq,
        #                                                                                                sequence_idx=
        #                                                                                             sentences_to_run[
        #                                                                                                    j])
        # random_cnt_list_moderate.append(random_cnt_moderate)
        # random_time_for_attack_list.append(random_time_for_attack)
        # atten_cnt_moderate, atten_cant_untoxic, atten_time_for_attack = attacker.attack_until_break(model='atten',
        #                                                                                             seq=seq,
        #                                                                                             sequence_idx=
        #                                                                                          sentences_to_run[j])
        # atten_cnt_list_moderate.append(atten_cnt_moderate)
        # atten_time_for_attack_list.append(atten_time_for_attack)
        hotflip_beam10_cnt_moderate, hotflip_beam10_time_for_attack   = attacker.attack_hot_flip_until_break(seq=seq,
                                                                        beam_size=10,  sequence_idx=sentences_to_run[j])

        hotflip_beam10_cnt_list_moderate.append(hotflip_beam10_cnt_moderate)
        hotflip_beam10_time_for_attack_list.append(hotflip_beam10_time_for_attack)


        hotflip_beam5_cnt_moderate, hotflip_beam5_time_for_attack   = attacker.attack_hot_flip_until_break(seq=seq,
                                                                        beam_size=5,  sequence_idx=sentences_to_run[j])
        hotflip_beam5_cnt_list_moderate.append(hotflip_beam5_cnt_moderate)
        hotflip_beam5_time_for_attack_list.append(hotflip_beam5_time_for_attack)

        hotflip_cnt_moderate, _, hotflip_time_for_attack = attacker.attack_until_break(model=
                                                                                       'hotflip',
                                                                                       seq=seq,
                                                                                       sequence_idx=sentences_to_run[j])

        hotflip_cnt_list_moderate.append(hotflip_cnt_moderate)
        hotflip_time_for_attack_list.append(hotflip_time_for_attack)


        detector_cnt_moderate, _, detector_time_for_attack = attacker.attack_until_break(
            model='detector',
            seq=seq,
            sequence_idx=sentences_to_run[j])
        detector_cnt_list_moderate.append(detector_cnt_moderate)
        detector_time_for_attack_list.append(detector_time_for_attack)

        ## Attack with hotflip, no smart
        attacker.config.smart_replace = False

        # hotflip_cnt_no_smart, _, hotflip_time_for_attack = attacker.attack_until_break(model=
        #                                                                                'hotflip',
        #                                                                                seq=seq,
        #                                                                              sequence_idx=sentences_to_run[j])
        # hotflip_cnt_list_no_smart.append(hotflip_cnt_no_smart)

        ## Attack all models, hard restrictions
        # attacker.config.smart_replace = True
        # attacker.config.flip_middle_letters_only = True
        # attacker.config.flip_once_in_a_word = True

        # random_cnt_hard, random_cant_untoxic, random_time_for_attack = attacker.attack_until_break(model='random',
        #                                                                                            seq=seq,
        #                                                                                            sequence_idx=
        #                                                                                          sentences_to_run[j])
        # random_cnt_list_hard.append(random_cnt_hard)
        # random_time_for_attack_list.append(random_time_for_attack)
        # atten_cnt_hard, atten_cant_untoxic, atten_time_for_attack = attacker.attack_until_break(model='atten',
        #                                                                                         seq=seq,
        #                                                                                         sequence_idx=
        #                                                                                         sentences_to_run[j])
        # atten_cnt_list_hard.append(atten_cnt_hard)
        # atten_time_for_attack_list.append(atten_time_for_attack)
        # hotflip_cnt_hard, _, hotflip_time_for_attack = attacker.attack_until_break(model=
        #                                                                            'hotflip',
        #                                                                            seq=seq,
        #                                                                            sequence_idx=sentences_to_run[j])
        # hotflip_cnt_list_hard.append(hotflip_cnt_hard)
        # hotflip_time_for_attack_list.append(hotflip_time_for_attack)
        # detector_cnt_hard, detector_cant_untoxic, detector_time_for_attack = attacker.attack_until_break(
        #     model='detector',
        #     seq=seq,
        #     sequence_idx=
        #     sentences_to_run[
        #         j])
        #
        # detector_cnt_list_hard.append(detector_cnt_hard)
        # detector_time_for_attack_list.append(detector_time_for_attack)
        # detector_cant_untoxic_cnt += detector_cant_untoxic
        # random_cant_untoxic_cnt += random_cant_untoxic
        # atten_cant_untoxic_cnt += atten_cant_untoxic
        # print("Random Cnt moderate: ", random_cnt_moderate)
        # print("Attention Cnt  moderate: ", atten_cnt_moderate)
        print("Hotflip Cnt  moderate: ", hotflip_cnt_moderate)
        print("Detector Cnt  moderate: ", detector_cnt_moderate)
        print("Hotflip beam5 Cnt  moderate: ", hotflip_beam5_cnt_moderate)
        print("Hotflip beam10 Cnt  moderate: ", hotflip_beam10_cnt_moderate)
        # print("Random Cnt hard: ", random_cnt_hard)
        # print("Attention Cnt  hard: ", atten_cnt_hard)
        # print("Hotflip Cnt  hard: ", hotflip_cnt_hard)
        # print("Detector Cnt  hard: ", detector_cnt_hard)
        #print("Hotflip no smart Cnt: ", hotflip_cnt_no_smart)
        if j % 100 == 99:
            np.save(path.join(RES_OUT_DIR, 'attack_dict.npy'), attacker.attack_list)

    print("Random mean moderate: ", np.mean(random_cnt_list_moderate))
    print("Random mean hard: ", np.mean(random_cnt_list_hard))
    print("Num of sentences cant untoxic random: ", str(random_cant_untoxic_cnt))
    print("Attention mean moderate: ", np.mean(atten_cnt_list_moderate))
    print("Attention mean hard: ", np.mean(atten_cnt_list_hard))
    print("Num of sentences cant untoxic attention: ", str(atten_cant_untoxic_cnt))
    print("Hotflip mean moderate: ", np.mean(hotflip_cnt_list_moderate))
    print("Hotflip beam 5 mean moderate: ", np.mean(hotflip_beam5_cnt_list_moderate))
    print("Hotflip beam 10 mean moderate: ", np.mean(hotflip_beam10_cnt_list_moderate))
    print("Hotflip mean hard: ", np.mean(hotflip_cnt_list_hard))
    print("Hotflip mean no smart: ", np.mean(hotflip_cnt_list_no_smart))
    print("Hotflip time for attack: ", np.mean(np.concatenate(hotflip_time_for_attack_list)))
    print("Detector mean moderate: ", np.mean(detector_cnt_list_moderate))
    print("Detector mean hard: ", np.mean(detector_cnt_list_hard))
    print("Num of sentences cant untoxic detector: ", str(detector_cant_untoxic_cnt))
    print("Detector time for attack: ", np.mean(np.concatenate(detector_time_for_attack_list)))
    print("Hotflip beam 5 time for attack: ", np.mean(np.concatenate(hotflip_beam5_time_for_attack_list)))
    print("Hotflip beam 10 time for attack: ", np.mean(np.concatenate(hotflip_beam10_time_for_attack_list)))

    flips_cnt = dict()
    flips_cnt['random_moderate'] = random_cnt_list_moderate
    flips_cnt['hotflip_moderate'] = hotflip_cnt_list_moderate
    flips_cnt['hotflip_beam5_moderate'] = hotflip_beam5_cnt_list_moderate
    flips_cnt['hotflip_beam10_moderate'] = hotflip_beam10_cnt_list_moderate
    flips_cnt['detector_moderate'] = detector_cnt_list_moderate
    flips_cnt['atten_moderate'] = atten_cnt_list_moderate
    flips_cnt['random_hard'] = random_cnt_list_hard
    flips_cnt['hotflip_hard'] = hotflip_cnt_list_hard
    flips_cnt['hotflip_no_smart'] = hotflip_cnt_list_no_smart
    flips_cnt['detector_hard'] = detector_cnt_list_hard
    flips_cnt['atten_hard'] = atten_cnt_list_hard
    np.save(path.join(RES_OUT_DIR, 'flips_cnt.npy'), flips_cnt)
    np.save(path.join(RES_OUT_DIR, 'attack_dict.npy'), attacker.attack_list)


if __name__ == '__main__':
    example()
