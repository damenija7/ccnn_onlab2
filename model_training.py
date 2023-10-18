# This is a sample Python script.
import json
import os
import sys
from math import floor, ceil
from pydoc import locate
from datetime import datetime
from pprint import pprint
from typing import Optional, Type, Callable, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler, RandomSampler

from dataset_embedding_cacher.embedding_cacher import EmbeddingCacher
from model_training import config
from model_training.dataset.cc_prediction_dataset import CCPredictionDataset
from model_training.dataset.cc_prediction_dataset_force_residue import CCPredictionDatasetForceResidue
from model_training.dataset.cc_prediction_dataset_per_residue import CCPredictionDatasetPerResidue
from model_training.dataset.collator import raw_collator
from model_training.sampler.gumbel_max import GumbelMaxWeightedRandomSampler
from model_training.train.trainer import Trainer
from model_training.visualization.visualize_training_results import visualize_training_results


# Import Any Class
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def run_training(
        dataset_path: str,
        dataset_validation_path: Optional[str],
        dataset_cache: Optional[str],
        num_epochs: int,
        batch_size: int,
        embedder_class_path,
        model_class_path,
        max_seq_len: int,
        k_fold_cross_validation: Optional[int],
        weighted_random_sampler: bool,
        force_per_residue: bool
):
    generator = torch.Generator().manual_seed(42)

    if model_class_path is None:
        raise Exception("Model not specified")

    print(
        f"Running Config: '{num_epochs}' Epochs, '{batch_size}' batch size, max sequence length:{max_seq_len}"
    )


    print("Initializing Embedder")
    # Clear GPU cache
    try:
        print(f"Using Device: {torch.cuda.get_device_name(config.Config.device)}")
        torch.cuda.empty_cache()
        memory_stats = torch.cuda.memory_stats()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
        print(f"GPU Total Mem: {total_memory / (1024 * 1024 * 1024)} GB, Usable: {available_memory / (1024 * 1024 * 1024)} GB ")
    except:
        pass

    embedder_class = locate(embedder_class_path)
    if embedder_class is None:
        raise Exception(f"Couldn't load embedder '{embedder_class_path}'")


    embedder = embedder_class()
    embedder.to(config.Config.device )
    embedder_type = embedder.__class__.__name__
    print(type(embedder))

    if dataset_cache is not None:
        cacher = EmbeddingCacher(sequence_embedder=embedder, cache_path=dataset_cache)
        # Use cacher as wrapper for embedder if cache is specified
        embedder = cacher

    print("Initializing Model")
    # Type == Class
    model_class: Optional[Type] = locate(model_class_path)
    if model_class is None:
        raise Exception(f"Couldn't load model '{model_class_path.split('.')[-1]}'")

    # number of channels embedder gives
    # pass as input to model constructor if possible
    in_channels = embedder.embed('AAA')[0].shape[-1]
    try:
        model = model_class(in_channels)
    except:
        model = model_class()

    model_type = model.__class__.__name__
    print(type(model))
    model.to(config.Config.device)


    print("Loading Datasets...")
    # CCPredictionDatasetPerResidue: ("uniprot_id", "residue_idx", "embedding", "label") header
    # CCPredictionDataset: ("uniprot_id","sequence","label") header
    dataset_class = get_dataset_class(dataset_path, force_per_residue)


    # TMP
    # TEST BEG, IN, END LABELS INSTEAD
    def label_transform_beginend(label) -> torch.Tensor:
        res = [float(digit) for digit in label]

        res_tensor = torch.zeros(size=(len(label), 3), dtype=torch.double)

        in_ = False
        for i in range(len(res) - 1):
            if res[i] == 1.0:
                if not in_:
                    # BEG
                    res_tensor[i, 0] = 1.0
                    in_ = True
                else:
                    # INSIDE
                    if res[i+1] == 1.0:
                        res_tensor[i, 2] = 1.0
                    # END
                    else:
                        res_tensor[i, 1] = 1.0
                        in_ = False
        if in_:
            res_tensor[len(res) - 1, 2] = 1.0

        return res_tensor[:, :2]





    train_dataset = dataset_class(
        csv_file_path=dataset_path,
        id_sequence_label_idx=0,
        max_seq_len=max_seq_len,
        # label_transform=label_transform_begin
        seq_embedder=embedder

    )

    if dataset_validation_path:
        val_dataset = get_dataset_class(dataset_validation_path, force_per_residue=False)(
            csv_file_path=dataset_validation_path,
            id_sequence_label_idx=0,
            max_seq_len=max_seq_len,
            # label_transform=label_transform_beginend,
            seq_embedder=embedder
        )
    else:
        val_dataset = None



    if not k_fold_cross_validation:
        print("Starting Normal Training")
        train_normal(batch_size=batch_size,
                     train_dataset=train_dataset,
                     val_dataset=val_dataset,
                     embedder=embedder, generator=generator, model=model,
                     model_type=model_type,
                     num_epochs=num_epochs, weighted_random_sampler=weighted_random_sampler)
    else:
        print(f"Starting K-Fold Training (K = {k_fold_cross_validation}")
        train_kfold(batch_size=batch_size,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    k_fold_cross_validation=k_fold_cross_validation,
                    embedder=embedder,
                    generator=generator,
                    max_seq_len=max_seq_len,
                    model=model,
                    model_type=model_type,
                    num_epochs=num_epochs, weighted_random_sampler=weighted_random_sampler)


def get_dataset_class(dataset_path, force_per_residue):
    with open(dataset_path) as f:
        if len(f.readline().strip().split(',')) == 3:
            if not force_per_residue:
                dataset_class = CCPredictionDataset
            else:
                dataset_class = CCPredictionDatasetForceResidue
        else:
            dataset_class = CCPredictionDatasetPerResidue
    return dataset_class


def train_normal(batch_size,
                 train_dataset: Union[CCPredictionDataset, CCPredictionDatasetPerResidue],
                 val_dataset: Optional[Union[CCPredictionDataset, CCPredictionDatasetPerResidue]],
                 embedder, generator, model, model_type,
                 num_epochs, weighted_random_sampler: bool):
    if val_dataset is None:
        train_dataset_type = type(train_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [floor(0.7 * len(train_dataset)), ceil(0.3 * len(train_dataset))],
                                                  generator=generator)
        train_dataset_type._set_attributes(train_dataset)
        train_dataset_type._set_attributes(val_dataset)



    print("Setting up Dataloader...")

    train_sampler, val_sampler = RandomSampler(train_dataset), RandomSampler(val_dataset)
    if weighted_random_sampler:
        train_sampler = get_weighted_train_sampler(train_dataset, train_sampler)
        #
        # Don't use oversampling for validation, use as is!
        # ( Would lead to false validation statitics )
        #val_weights = [val_dataset.pos_rate if label < 1 else 1.0 - val_dataset.pos_rate for label in val_dataset.labels]
        #val_sampler = WeightedRandomSampler(weights=val_weights, num_samples=len(val_dataset), replacement=True)

    train_dataloader, val_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=raw_collator, sampler=train_sampler, drop_last=True
    ), DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=raw_collator, sampler=val_sampler, drop_last=True
    )
    trainer = Trainer(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        weighted_random_sampler=weighted_random_sampler
    )
    # Output all results to folder workdir->training_results_path
    training_results_path = f"output/training_results_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)
    # Write training parameters
    with open(f"{training_results_path}/parameters.json", "w") as f:
        json.dump(vars(args), f)
    training_results = trainer.train(
        model=model,
        embedder=embedder,
        num_epochs=num_epochs,
        best_model_save_file_path=f"{training_results_path}/best_model_{model_type}.pkl",
    )
    results = {'training_results': training_results}
    results['model_type'] = model_type
    with open(f"{training_results_path}/results.json", "w") as f:
        json.dump(results, f)
    visualize_training_results(results, training_results_path + "/result")


def get_weighted_train_sampler(train_dataset, train_sampler):

    if (isinstance(train_dataset.labels[0], torch.Tensor) or isinstance(train_dataset.labels[0], np.ndarray)):
        if torch.numel(next(iter(train_dataset))[1]) == 1:
            train_weights = [train_dataset.pos_rate if pos < 1 else 1.0 - train_dataset.pos_rate for pos in train_dataset.pos]
            train_weights_sum = sum(train_weights)
            train_weights = [weight / train_weights_sum for weight in train_weights]
            train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)
        else:

            train_weights = []
            for label_idx, (_, label) in enumerate(train_dataset):
                #logit = torch.numel(label[label < 1]) * np.log(train_dataset.pos_rate)
                logit = train_dataset.pos[label_idx] * np.log(train_dataset.pos_rate)
                #logit += torch.numel(label[label > 0]) * np.log(1.0 - train_dataset.pos_rate)
                logit += (torch.numel(label) - train_dataset.pos[label_idx]) * np.log(1.0 - train_dataset.pos_rate)


                # (length'th root is 1/length in logit space)
                logit /= torch.numel(label)
                # weight = np.exp(weight)
                train_weights.append(np.exp(logit))
            train_weights_sum = sum(train_weights)
            train_weights = [weight / train_weights_sum for weight in train_weights]

            # train_sampler = GumbelMaxWeightedRandomSampler(logits=train_weights, num_samples=len(train_dataset))
            train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)

    else:
        train_weights = [train_dataset.pos_rate if label < 1 else 1.0 - train_dataset.pos_rate for _, label in
                         train_dataset]
        train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)
    return train_sampler


def train_kfold(batch_size: int,
                train_dataset: Union[CCPredictionDataset, CCPredictionDatasetPerResidue],
                val_dataset: Optional[Union[CCPredictionDataset, CCPredictionDatasetPerResidue]],
                k_fold_cross_validation: int, embedder: Callable, generator, max_seq_len: int,
                model: torch.nn.Module, model_type: str,
                    num_epochs: int,
                weighted_random_sampler: bool):

    dataset = train_dataset

    # In KFold mode Treat val dataset as just another part of whole dataset ( concat )
    if val_dataset:
        assert type(train_dataset) == type(val_dataset)
        dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
        # So that original type can be inferred during to train to embed batch sequences when reuqired
        dataset.dataset = train_dataset
        type(train_dataset)._set_attributes(dataset)


    # Output all results to folder workdir->training_results_path
    training_results_path = f"output/training_results_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    if not os.path.exists(training_results_path):
        os.makedirs(training_results_path)
    # Write training parameters
    with open(f"{training_results_path}/parameters.json", "w") as f:
        parameters = vars(args)
        parameters['model_num_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        json.dump(vars(args), f)



    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=k_fold_cross_validation, shuffle=True)

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        fold_train_dataset, fold_val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
        # So that original type can be inferred during to train to embed batch sequences when reuqired
        fold_train_dataset.dataset, fold_val_dataset = dataset, dataset

        type(train_dataset)._set_attributes(fold_train_dataset)
        type(train_dataset)._set_attributes(fold_val_dataset)

        fold_train_sampler, fold_val_sampler = RandomSampler(fold_train_dataset), RandomSampler(fold_val_dataset)

        if weighted_random_sampler:
                fold_train_sampler = get_weighted_train_sampler(fold_train_dataset, fold_train_sampler)


        train_dataloader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size,
                                                   sampler=fold_train_sampler,
                                                       collate_fn=raw_collator,
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size,
                                                   sampler=fold_val_sampler,
                                                     collate_fn=raw_collator,
                                                     drop_last=True)

        trainer = Trainer(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
        )


        best_model_save_file_path = f"{training_results_path}/best_model_{model_type}.pkl"


        training_results = trainer.train(
            model=model,
            embedder=embedder,
            num_epochs=ceil(num_epochs / k_fold_cross_validation),
            best_model_save_file_path=best_model_save_file_path,
        )
        model = torch.load(best_model_save_file_path)

        results = {'training_results': training_results}
        results['model_type'] = model_type
        with open(f"{training_results_path}/results_k_fold_{fold_idx}.json", "w") as f:
            json.dump(results, f)
        visualize_training_results(results, training_results_path + f"/results_k_fold_{fold_idx}")






# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # config.Config.device = torch.device('vulkan')\

    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--epochs", default=10)
    parser.add_argument("-b", "--batch_size", default=16)
    # Model to use
    parser.add_argument("-m", "--model", required=True)
    # Embedder to use
    parser.add_argument("-emb", "--embedder", required=True)

    # Dataset to train and validate on ( 7:3 split if dataset_validation_input not specified
    parser.add_argument("-i", "--dataset_input", required=True)
    parser.add_argument("-iv", "--dataset_validation_input", default=None)

    parser.add_argument("-c", "--dataset_cache", required=False)
    # Device to use
    parser.add_argument("-d", "--device", default='cpu')

    parser.add_argument("-msl", "--max_sequence_length", default=sys.maxsize)


    # If active use k-fold cross validation ( k value specifies how many folds to use)
    parser.add_argument("-k", "--k_fold_cross_validation", default=None)

    parser.add_argument('-wrs', "--weighted_random_sampler", default=False, action='store_true')

    # Gives residues instead of sequences but going through batch will be much slower during training
    parser.add_argument("-fpr", "--force_per_residue", default=False, action='store_true')

    args = parser.parse_args()

    pprint(vars(args))

    config.Config.device = torch.device(args.device)
    num_epochs, batch_size = int(args.epochs), int(args.batch_size)
    dataset_cache = args.dataset_cache if args.dataset_cache else None
    model = args.model
    embedder = args.embedder




    run_training(
        dataset_path=args.dataset_input,
        dataset_validation_path=args.dataset_validation_input,
        dataset_cache=dataset_cache,
        num_epochs=num_epochs,
        batch_size=batch_size,
        embedder_class_path=embedder,
        model_class_path=model,
        max_seq_len=int(args.max_sequence_length),
        k_fold_cross_validation=int(args.k_fold_cross_validation) if args.k_fold_cross_validation else None,
        weighted_random_sampler=args.weighted_random_sampler,
        force_per_residue=args.force_per_residue
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
