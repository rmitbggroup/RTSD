import os
import json
import numpy as np
import time
import argparse
from sklearn.metrics import pairwise_distances


MAX_DIST_FILE = "max_dist.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="time series dataset")
    return parser.parse_args()


def read_max_dist():
    """ read max pairwise distances from json file """

    if os.path.exists(MAX_DIST_FILE):
        with open(MAX_DIST_FILE, "r") as file:
            max_dist_dict = json.load(file)
        return max_dist_dict
    else:
        return {}


def write_max_dist(max_dist_dict):
    """ update max pairwise distances in json file """

    with open(MAX_DIST_FILE, "w") as file:
        json.dump(max_dist_dict, file, indent=4)


if __name__ == "__main__":
    args = parse_args()

    # load time series data
    list_of_ts = np.load("data/{}.npy".format(args.dataset))

    t_start = time.time()

    # calculate pairwise distances (Euclidean distance) among time series
    dist_matrix = pairwise_distances(list_of_ts, metric="euclidean", n_jobs=-1)
    print("| Time used: {:.3f} s".format(time.time() - t_start))

    # read and update new max distance in json file
    max_dist_dict = read_max_dist()
    max_dist_dict[args.dataset] = np.max(dist_matrix)
    write_max_dist(max_dist_dict)
    
    # save pairwise distance matrix
    np.save("dist/{}.npy".format(args.dataset), dist_matrix)
