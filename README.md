# Representative Time Series Discovery (RTSD)

## Packages

1. python 3.8.0
2. torch 2.0.1
3. torchvision 0.15.2
4. numpy 1.24.4
5. scikit-learn 1.3.0

## Datasets

All datasets ([ECG](https://github.com/DSM-fudan/Dumpy?tab=readme-ov-file#datasets), [Seismic](https://ln5.sync.com/dl/0b8135230/39vxx8su-tkfi7t2s-dgsvh8rp-k8ixcs8p?sync_id=12175200430004), [SALD](https://ln5.sync.com/dl/0b8135230/39vxx8su-tkfi7t2s-dgsvh8rp-k8ixcs8p?sync_id=12175200420004), [DEEP](https://ln5.sync.com/dl/0b8135230/39vxx8su-tkfi7t2s-dgsvh8rp-k8ixcs8p?sync_id=12175200450004) and [METR-LA](https://github.com/liyaguang/DCRNN?tab=readme-ov-file#data-preparation)) are publicly available online. Please refer to the links provided to access them. Then, process the dataset into an array of time series in `.npy` format and place them in the `data/` directory. Due to the large size of the datasets, we provide an example of a small sampled dataset in the `data/` directory.

## Time Series Embedding

This repo uses [SEAnet](https://github.com/qtwang/SEAnet) to generate time series embeddings. Please refer to the link provided to generate the embeddings. Process the embeddings into an array of time series embeddings in `.npy` format (similar to dataset processing) and place them in the `emb/` directory.

## Directory Description

1. `data/`: A directory that stores time series for each dataset.
2. `dist/`: A directory that stores distance matrix each dataset.
3. `emb/`: A directory that stores time series embeddings for each dataset.
4. `models/`: A directory that stores trained models.
5. `simset/`: A directory that stores precomputed similar sets.

## File Description

1. `compute_dist.py`: A file that computes distance matrix for a dataset.
2. `max_dist.json`: A file that stores the maximum pairwise distance for each dataset.
3. `compute_simset.py`: A file that computes similar set for a dataset.
4. `greedy.py`: The implementation of *PreGreedy*, *PreGreedyET*, *Greedy* and *GreedyET* representative time series selection methods. This file reads the input dataset, selection method, normalized distance threshold and coverage threshold, and outputs the representative time series selected.
5. `mlp.py`: A file that defines the neural network.
6. `mlgreedyet_train.py`: A file that generates the training data for training the neural network.
7. `mlgreedyet_test.py`: The implementation of *MLGreedyET* representative time series selection method. This file reads the input dataset, normalized distance threshold and coverage threshold, and outputs the representative time series selected.

## Non-Learning Based Selection

For representative time series selection **without** similar sets precomputation (*Greedy* and *GreedyET*), please refer to steps 1 and 3. For representative time series selection **with** similar sets precomputation (*PreGreedy* and *PreGreedyET*), please refer to steps 1, 2 and 3. Note that for each dataset, steps 1 and 2 are required to run only once initially.

1. Compute distance matrix: `python compute_dist.py --dataset <dataset_filename>`
2. Compute similar set: `python compute_simset.py --dataset <dataset_filename> --tau <normalized_distance_threshold>`
3. Select representative time series: `python greedy.py --dataset <dataset_filename> --method <selection_method> --tau <normalized_distance_threshold> --beta <coverage_threshold>`

## Learning Based Selection

For representative time series selection using learning approach (*MLGreedyET*), please refer to each step below.

1. Train the model: `python mlgreedyet_train.py --dataset <dataset_filename> --tau <training_normalized_distance_threshold>`
2. Select representative time series: `python mlgreedyet_test.py --dataset <dataset_filename> --tau <normalized_distance_threshold> --beta <coverage_threshold>`
