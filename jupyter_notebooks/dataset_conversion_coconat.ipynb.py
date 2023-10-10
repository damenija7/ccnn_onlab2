#%%
# https://coconat.biocomp.unibo.it/datasets/
import csv
from tqdm import tqdm

training_pos_path = '../input/training.positive.annot.fasta'
training_neg_path = '../input/training.negative.fasta'
test_pos_path = '../input/dataset_718.biounit.positive.annot.fasta'
test_neg_path = '../input/dataset_718.negative.fasta'

def get_rec_data(rec):
    uniprot_id = rec.id
    seq_len = 0
    char: str
    for char in rec.seq:
        if char.isalpha() and char.isupper():
            seq_len += 1
        else:
            break
    seq = rec.seq[:seq_len]
    label = ''.join(['0' if char == 'i' else '1' for char in rec.seq[seq_len:2 * seq_len]])

    return uniprot_id, seq, label


#%%
from Bio import SeqIO

#input
training_pos = SeqIO.parse(training_pos_path, 'fasta')
training_neg = SeqIO.parse(training_neg_path, 'fasta')
test_pos = SeqIO.parse(test_pos_path, 'fasta')
test_neg = SeqIO.parse(test_neg_path, 'fasta')

# output
training_id_to_seq_label = {}
test_id_to_seq_label = {}


# get output
for i in range(10):
    cross_valid_set_path = f'../input/cv_sets/set{i}'
    cross_valid_sets = [line.strip() for line in cross_valid_set_path if len(line.strip()) > 0]

for records in [training_pos, training_neg]:
    for rec in records:
        uniprot_id, seq, label = get_rec_data(rec)
        training_id_to_seq_label[uniprot_id] = (seq, label)
for records in [test_pos, test_neg]:
    for rec in records:
        uniprot_id, seq, label = get_rec_data(rec)
        test_id_to_seq_label[uniprot_id] = (seq, label)


with open('../input/coconat_training.csv', "w") as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(('uniprot_id', 'seq', 'label'))
    for uniprot_id in tqdm(training_id_to_seq_label.keys(), total=len(training_id_to_seq_label)):
        seq, label = training_id_to_seq_label[uniprot_id]
        csv_writer.writerow((uniprot_id, seq, label))



# Training
with open('../input/coconat_test.csv', "w") as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(('uniprot_id', 'seq', 'label'))
    for uniprot_id in tqdm(test_id_to_seq_label.keys(), total=len(test_id_to_seq_label)):
        seq, label = test_id_to_seq_label[uniprot_id]
        csv_writer.writerow((uniprot_id, seq, label))

