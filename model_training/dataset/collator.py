# Does nothing with the sequences and labels and skips stacking sequences as if they were a tensor
def raw_collator(sequences_labels):
    sequences, labels = list(zip(*sequences_labels))
    return sequences, labels
