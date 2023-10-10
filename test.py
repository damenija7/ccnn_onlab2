from model_training.dataset.cc_prediction_dataset_per_residue import CCPredictionDatasetPerResidue

ds = CCPredictionDatasetPerResidue('input/input_own.processed.csv')



print(ds[0])
