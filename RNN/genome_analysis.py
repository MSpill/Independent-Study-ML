import numpy as np
import rnn
import data.onehotgenome as one_hotter
import data.onehottext as text_hotter
import pickle

# program to check RNN's genome generation against real data
onehot_data = one_hotter.one_hot_genome(
    "/Users/matthewspillman/Documents/_12th/Indep Study/Independent-Study-ML/RNN/data/genome.fna")
charset = onehot_data[0]
one_hots = onehot_data[1]

print("loaded genome, sampling text...")
genome_rnn = pickle.load(open('rnn9.rnn', 'rb'))
genome_rnn.reset_state()

temp = 0.9
sample = genome_rnn.sample_text(charset, temp, 100000)
print("calculating frequencies")

# hardcoded this from a calculation
real_freq = [0.24016074049075736, 0.25670782961711164,
             0.24183299527372668, 0.2612984346184043]

gen_freq = [0, 0, 0, 0]

for i in range(0, len(sample)):
    for j in range(4):
        if sample[i] == charset[j]:
            gen_freq[j] += 1.0

for i in range(4):
    gen_freq[i] /= len(sample)

# regardless of sampling temperature, the frequencies don't match
# still, the predicted probabilities before sampling often match the real frequencies
# the discrepancy is likely because the RNN latched onto some patterns
# like AAA and TTT which skew total frequency
print(real_freq)
print(gen_freq)

'''
freq = [0, 0, 0, 0]
for i in range(0, len(one_hots)):
    for j in range(0, 4):
        if one_hots[i][j] == 1:
            freq[j] += 1.0

print(charset)
print(freq)
for i in range(4):
    freq[i] /= len(one_hots)
print(freq)
I saved the output and hardcoded it
'''

'''genome_str = ""
for i in range(0, 10000):
    picked_index = -1
    for j in range(0, len(one_hots[i])):
        if one_hots[i][j] == 1:
            picked_index = j
    genome_str = "{0}{1}".format(genome_str, charset[picked_index])

print(genome_str)
'''
