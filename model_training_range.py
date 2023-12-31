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
from model_training.dataset.cc_prediction_dataset_range import CCPredictionDatasetRange
from model_training.dataset.collator import raw_collator
from model_training.train.trainer import Trainer
from model_training.train.trainer_range import TrainerRange
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
        **kwargs
):
    generator = torch.Generator().manual_seed(42)

    if model_class_path is None:
        raise Exception("Model not specified")

    print(
        f"Running Config: '{num_epochs}' Epochs, '{batch_size}' batch size, max sequence length:{max_seq_len}"
    )

    #
    # 1.
    #
    print("Initializing Embedder")
    # Clear GPU cache
    try:
        print_gpu_stats()
    except:
        pass


    def embedder(x):
        raise Exception("No embedder")

    try:
        embedder_class = load_class(embedder_class_path)
        embedder = embedder_class()
        embedder.to(config.Config.device)
        print(type(embedder))
    except:
        pass
  

    if dataset_cache is not None:
        cacher = EmbeddingCacher(sequence_embedder=embedder, cache_path=dataset_cache)
        # Use cacher as wrapper for embedder if cache is specified
        embedder = cacher

    #
    # 2.
    #
    print("Initializing Model")
    model_class: Optional[Type] = load_class(model_class_path)

    # number of channels embedder gives
    # pass as input to model constructor if possible
    in_channels = embedder.embed('AAA')[0].shape[-1]
    try:
        model = model_class(in_channels).to(config.Config.device)
    except Exception as e:
        model = model_class().to(config.Config.device)
    print(type(model))


    print("Loading Datasets...")


    train_dataset = CCPredictionDatasetRange(
        csv_file_path=dataset_path,
        id_sequence_label_idx=0,
        max_seq_len=max_seq_len,
        # label_transform=label_transform_begin

    )

    if dataset_validation_path:
        val_dataset = CCPredictionDatasetRange(
            csv_file_path=dataset_validation_path,
            id_sequence_label_idx=0,
            max_seq_len=max_seq_len,
            # label_transform=label_transform_beginend,
        )
    else:
        val_dataset = None



    if not k_fold_cross_validation:
        print("Starting Normal Training")
        train_normal(batch_size=batch_size,
                     train_dataset=train_dataset,
                     val_dataset=val_dataset,
                     embedder=embedder, generator=generator, model=model,
                     num_epochs=num_epochs, weighted_random_sampler=weighted_random_sampler)
    else:
        print(f"Starting K-Fold Training (K = {k_fold_cross_validation}")
        train_kfold(batch_size=batch_size,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    k_fold_cross_validation=k_fold_cross_validation,
                    embedder=embedder,
                    model=model,
                    num_epochs=num_epochs, weighted_random_sampler=weighted_random_sampler)


def load_class(class_path) -> Type:
    embedder_class = locate(class_path)
    if embedder_class is None:
        raise Exception(f"Couldn't load class '{class_path}'")
    return embedder_class


def print_gpu_stats():
    print(f"Using Device: {torch.cuda.get_device_name(config.Config.device)}")
    torch.cuda.empty_cache()
    memory_stats = torch.cuda.memory_stats()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
    print(
        f"GPU Total Mem: {total_memory / (1024 * 1024 * 1024)} GB, Usable: {available_memory / (1024 * 1024 * 1024)} GB ")

def get_weighted_train_sampler(train_dataset):
    train_weights = []

    for seq, label in train_dataset:
        pos_seq = sum(bbox[1] for bbox in label)
        n_seq = 1.0 - pos_seq

        weight = pos_seq * np.log(max(train_dataset.pos_rate, 1e-8)) + n_seq * np.log(max(1 - train_dataset.pos_rate, 1e-8))
        weight = np.exp(weight)

        train_weights.append(weight)



    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset), replacement=True)
    return train_sampler


def train_normal(batch_size,
                 train_dataset: CCPredictionDatasetRange,
                 val_dataset: Optional[CCPredictionDatasetRange],
                 embedder, generator, model,
                 num_epochs, weighted_random_sampler: bool):

    model_type = model.__class__.__name__

    # do 7:3 train val split if separate val dataset not specified
    if val_dataset is None:
        train_dataset_type = type(train_dataset)
        train_dataset, val_dataset = random_split(train_dataset, [floor(0.7 * len(train_dataset)), ceil(0.3 * len(train_dataset))],
                                                  generator=generator)
        train_dataset_type._set_attributes(train_dataset)
        train_dataset_type._set_attributes(val_dataset)



    print("Setting up Dataloader...")

    train_sampler, val_sampler = RandomSampler(train_dataset), RandomSampler(val_dataset)
    if weighted_random_sampler:
        train_sampler = get_weighted_train_sampler(train_dataset)
        # Don't use oversampling for validation, use as is!
        # ( Would lead to false validation statitics )


    train_dataloader= DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=raw_collator, sampler=train_sampler, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=raw_collator, sampler=val_sampler, drop_last=True
    )
    trainer = TrainerRange(
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




def train_kfold(batch_size: int,
                train_dataset:CCPredictionDatasetRange,
                val_dataset: Optional[CCPredictionDatasetRange],
                k_fold_cross_validation: int, embedder: Callable,
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
                fold_train_sampler = get_weighted_train_sampler(fold_train_dataset)


        train_dataloader = torch.utils.data.DataLoader(fold_train_dataset, batch_size=batch_size,
                                                   sampler=fold_train_sampler,
                                                       collate_fn=raw_collator,
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(fold_val_dataset, batch_size=batch_size,
                                                   sampler=fold_val_sampler,
                                                     collate_fn=raw_collator,
                                                     drop_last=True)

        trainer = TrainerRange(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            weighted_random_sampler=False
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
        weighted_random_sampler=args.weighted_random_sampler
    )

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
