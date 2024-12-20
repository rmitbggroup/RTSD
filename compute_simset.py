import numpy as np
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="time series dataset")
    parser.add_argument("--tau", type=float, required=True, help="normalized distance threshold between [0, 1]")
    return parser.parse_args()


def load_dist_matrix(args):
    """ load distance matrix """

    dist_matrix = np.load("dist/{}.npy".format(args.dataset))
    max_dist = np.max(dist_matrix)

    return dist_matrix, max_dist


def compute_simset(args, dist_matrix, max_dist):
    """ compute the similar set for each time series """

    list_of_simsets = []
    tau = args.tau * max_dist

    for dist_list in dist_matrix:
        simset = set()
        for ts_id, dist in enumerate(dist_list):
            if dist <= tau:
                simset.add(ts_id)
        list_of_simsets.append(simset)
    
    return list_of_simsets


if __name__ == "__main__":
    args = parse_args()

    dist_matrix, max_dist = load_dist_matrix(args)
    
    t_start = time.time()
    list_of_simsets = compute_simset(args, dist_matrix, max_dist)

    print("| Time used: {:.3f} s".format(time.time() - t_start))

    # save similar sets
    np.save("simset/{}_t{}.npy".format(args.dataset, args.tau), list_of_simsets)
