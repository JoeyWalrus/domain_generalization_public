# domain_generalization_public
Code repository to master thesis "Domain Generalization by Deep Learning-Based Domain Processing"

This repository contains all the code used to create the results presented in my master's thesis, "Domain Generalization by Deep Learning-Based Domain Encoding".
In the following, a short description of the individual files is outlined:

The files starting with "preprocessing_part_*" contain preprocessing steps for the individual data sets with the main task of bringing the different data sets into a common form. Also, additional preprocessing steps like normalization, etc., can be found here.

The entire training and evaluation routine is distributed across the following files:
cv_fold_part | returns the indices for the individual cross-validation splits
kde_part | computes the kernel density estimation for each feature and returns domain embeddings for each domain in the same shape as the input data
models | contains all the embedding heads and the evaluation head + the architecture combining both (the only model class needed for training all configurations)
prepare_training_part | routine that prepares data loaders and everything else needed for training the model
training_part | holds the actual training routine
eval_part | contains all the functions for evaluating the model's predictions

The files ending with "dataset_evaluation" contain all the experiments carried out that led to the results presented in the thesis. Most results were also stored in respective .npy files to be accessed without having to rerun the entire process.

Unfortunately, the data sets are too large to be uploaded here, but the synthetic data set is generated in the respecting preprocessing file, and the other two data sets are publicly available (links can be found in the thesis).


