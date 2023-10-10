import csv
import logging
import sys
from pprint import pprint
from pydoc import locate

import torch
from tqdm import tqdm

from dataset_embedding_cacher.embedding_cacher import EmbeddingCacher


# Import Any Class
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--embedder", required=True)
    parser.add_argument("-d", "--device", default='cpu')
    parser.add_argument("-msl", "--max_sequence_length", default=sys.maxsize)

    args = parser.parse_args()

    pprint(vars(args))

    input_path, output_path = args.input, args.output
    max_sequence_length: int = int(args.max_sequence_length)

    embedder_class_path = args.embedder


    embedder_class = locate(embedder_class_path)
    if embedder_class is None:
        raise Exception(f"Couldn't load embedder '{embedder_class_path}'")

    embedder = embedder_class()
    try:
        embedder.to(torch.device(args.device))
    except:
        pass

    embedding_cacher = EmbeddingCacher(sequence_embedder=embedder, cache_path=output_path)

    processed_input_path = input_path[:input_path.rindex(".csv")] + ".processed.csv"



    with open(processed_input_path, "w") as f_processed_input:
        processed_input_csv_writer = csv.writer(f_processed_input, delimiter=',')
        processed_input_csv_writer.writerow(("uniprot_id", "residue_idx", "embedding", "label"))

        logging.info(f"Starting Embedding of '{input_path}'...")
        with open(input_path) as f_input:
            csv_reader = csv.reader(f_input, delimiter=',')

            # Skip Header
            next(csv_reader)


            # uniprot accession IDs
            uniprot_ids = []
            sequences = []
            labels = []

            for row in csv_reader:

                #uniprot id
                uniprot_id = row[0]
                sequence = row[1]
                label = row[2]

                if len(sequence) <= max_sequence_length:
                    sequences.append(sequence)
                    uniprot_ids.append(uniprot_id)
                    labels.append(label)

        for sequence, uniprot_id, seq_label in tqdm(zip(sequences, uniprot_ids, labels), desc="Caching Embeddings...", total=len(sequences)):
            torch.cuda.empty_cache()
            embedding_cacher.cache_embeddings([sequence])
            embedding = embedding_cacher.get_embeddings([sequence])[0]

            # per-residue format
            for res_idx in range(len(sequence)):
                res_embedding = embedding[res_idx]
                res_embedding = str([x.item() for x in res_embedding])
                res_label = seq_label[res_idx]
                # uniprot id, residue idx, embedding, label
                processed_input_csv_writer.writerow((uniprot_id, res_idx, res_embedding, res_label))


