import numpy as np
import tqdm
import matplotlib.pyplot as plt

MAX_SEQ = 400

attacks = np.load('/home/yotamg/Project/toxic-fool-flip-detector/toxic-fool/toxic_fool/resources_out/attack_dict.npy')

random_attacks = list()

models = np.unique([a['attack_model'] for a in attacks ])

attacks_per_step = dict()
attacks_per_step_hard = dict()

tox_after_5 = dict()
tox_after_5_hard = dict()
num_of_sentences = dict()
num_of_sentences_hard = dict()
num_of_toxic_sentences_after_5_replacements = dict()
tox_after_replacment = dict()

time_for_attack_per_seq_length = dict()
num_of_flips_for_sentence = dict()
mean_num_of_flips = dict()
for model in models:
    attacks_per_step[model] = [0] * MAX_SEQ
    attacks_per_step_hard[model] = [0] * MAX_SEQ
    tox_after_replacment[model] = [0] * MAX_SEQ
    time_for_attack_per_seq_length[model] = dict()
    for i in range(1, MAX_SEQ):
        time_for_attack_per_seq_length[model][i] = list()
    num_of_flips_for_sentence[model] = list()
    mean_num_of_flips[model] = 0

attacks_per_step['hotflip_no_smart'] = [0] * MAX_SEQ

unique_sequences = np.unique([a['seq_idx'] for a in attacks if a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']])

## Find survival rate
for model in tqdm.tqdm(models):
    for i in tqdm.tqdm(range(1,MAX_SEQ)):
        survivals = len([a for a in attacks if a['attack_model'] == model and a['attack_number'] == i and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']])
        # tox_after_replacment[model][i-1] = np.mean([a['tox_after'] for a in attacks if a['attack_model'] == model and a['attack_number'] == i and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']])
        if survivals == 0:
            break
        attacks_per_step[model][i-1] = survivals
        # time_for_attack_per_seq_length[model][i] = [a['time_for_attack'] for a in attacks if a['attack_model'] == model and a['seq_length'] == i and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']]

for model in tqdm.tqdm(models):
    for i in tqdm.tqdm(range(1,MAX_SEQ)):
        survivals = len([a for a in attacks if a['attack_model'] == model and a['attack_number'] == i and a['smart_replace'] and a['flip_once_in_a_word'] and a['flip_middle_letters_only']])
        if survivals == 0:
            break
        attacks_per_step_hard[model][i - 1] = survivals

# plt.subplot(2,2,1)
plt.title("Percentage of Toxic Sentences")
for model in models:
    attacks_per_step[model] = (attacks_per_step[model] / np.max(attacks_per_step[model])) * 100
    plt.semilogx(attacks_per_step[model], label=model)
plt.ylabel("[%]")
plt.xlabel("Attack number")
plt.legend()

plt.title("Percentage of Toxic Sentences with Hard Restrictions")
for model in models:
    attacks_per_step_hard[model] = (attacks_per_step_hard[model] / np.max(attacks_per_step_hard[model])) * 100
    plt.semilogx(attacks_per_step_hard[model], label=model)
plt.ylabel("[%]")
plt.xlabel("Attack number")
plt.legend()

# plt.subplot(2, 2, 2)
# for model in models:
#     plt.plot(tox_after_replacment[model], label=model)
# plt.ylabel("Mean toxicity after X attacks")
# plt.xlabel("Attack number")
# plt.legend()
# plt.subplot(2, 2, 3)
# for model in models:
#     plt.plot(time_for_attack_per_seq_length[model], label=model)
# plt.ylabel("Time for attack")
# plt.xlabel("Sequence length")
# plt.legend()

## Find toxicity after 5 replacements
for model in tqdm.tqdm(models):
    num_of_sentences[model] = len(np.unique([a['seq_idx'] for a in attacks if a['attack_model'] == model and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']]))
    tox_after_5[model] = len([a for a in attacks if a['attack_model'] == model and a['attack_number'] == 5 and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only'] and a['tox_after'] > 0.5])

for model in tqdm.tqdm(models):
    num_of_sentences_hard[model] = len(np.unique([a['seq_idx'] for a in attacks if a['attack_model'] == model and a['smart_replace'] and a['flip_once_in_a_word'] and a['flip_middle_letters_only']]))
    tox_after_5_hard[model] = len([a for a in attacks if a['attack_model'] == model and a['attack_number'] == 5 and a['smart_replace'] and  a['flip_once_in_a_word'] and  a['flip_middle_letters_only'] and a['tox_after'] > 0.5])


n_groups = 4

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

succesful_attacks = list()
for model in models:
    succesful_attacks_for_model_percentage = ((num_of_sentences_hard[model] - tox_after_5_hard[model]) / num_of_sentences_hard[model])*100
    succesful_attacks.append(succesful_attacks_for_model_percentage)


rects1 = ax.bar(index, succesful_attacks,
                alpha=opacity, color='b')

ax.set_xlabel('Model')
ax.set_ylabel('[%]')
ax.set_title('Succesful Attacks with Hard Restrictions')
ax.set_xticks(index)
ax.set_xticklabels(['Attention', 'Detector', 'HotFlip','Random'])
ax.legend()
# fig.tight_layout()
plt.show()

for model in tqdm.tqdm(models):
    for seq in unique_sequences:
        num_of_flips_for_sentence[model].append(np.max([a['attack_number'] for a in attacks if a['attack_model'] == model and a['seq_idx'] == seq and a['smart_replace'] and not a['flip_once_in_a_word'] and not a['flip_middle_letters_only']]))
    mean_num_of_flips[model] = np.mean(num_of_flips_for_sentence[model])



print ("A")