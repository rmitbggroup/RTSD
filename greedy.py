import os
import json
import numpy as np
import time
import argparse
import heapq
from sklearn.metrics import pairwise_distances


MAX_DIST_FILE = "max_dist.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="time series dataset")
    parser.add_argument("--method", type=str, required=True, choices=["pregreedy", "pregreedyET", "greedy", "greedyET"], help="greedy selection method")
    parser.add_argument("--tau", type=float, required=True, help="normalized distance threshold between [0, 1]")
    parser.add_argument("--beta", type=float, default=1, help="coverage threshold between [0, 1]")
    return parser.parse_args()


def load_data(args):
    """ load time series data """

    list_of_ts = np.load("data/{}.npy".format(args.dataset))
    return list_of_ts


def load_simset(args):
    """ load similar sets """
    
    list_of_simsets = np.load("simset/{}_t{}.npy".format(args.dataset, args.tau), allow_pickle=True)
    return list_of_simsets


def read_max_dist():
    """ read max pairwise distances from json file """

    if os.path.exists(MAX_DIST_FILE):
        with open(MAX_DIST_FILE, "r") as file:
            max_dist_dict = json.load(file)
        return max_dist_dict
    else:
        return {}


def pregreedy(list_of_simsets, total_num_ts, num_ts_to_cover):
    """ use pre-computed similar sets + select representative time series using Greedy """
    
    # populate with time series indices
    set_of_all_ts = set(np.arange(total_num_ts))
    set_of_uncovered_ts = set(np.arange(total_num_ts))
    set_of_rep_ts = set()

    # select representative time series
    while total_num_ts - len(set_of_uncovered_ts) < num_ts_to_cover:
        max_gain = 0
        rep_ts = None
        
        for cand_ts in set_of_all_ts - set_of_rep_ts:
            cur_gain = len(list_of_simsets[cand_ts].intersection(set_of_uncovered_ts))

            if cur_gain > max_gain:
                max_gain = cur_gain
                rep_ts = cand_ts
        
        set_of_rep_ts.add(rep_ts)  # add representative time series with max marginal coverage
        set_of_uncovered_ts.difference_update(list_of_simsets[rep_ts])  # remove covered time series
    
    return set_of_rep_ts


def pregreedyET(list_of_simsets, total_num_ts, num_ts_to_cover):
    """ use pre-computed similar sets + select representative time series using Greedy with Early Termination """

    # populate with time series indices
    set_of_all_ts = set(np.arange(total_num_ts))
    set_of_uncovered_ts = set(np.arange(total_num_ts))
    set_of_rep_ts = set()

    pq = [] 

    # select representative time series
    while total_num_ts - len(set_of_uncovered_ts) < num_ts_to_cover:
        if not set_of_rep_ts:
            # initialize queue
            for cand_ts in set_of_all_ts:
                ub = len(list_of_simsets[cand_ts])  # marginal coverage of candidate time series
                heapq.heappush(pq, (-ub, cand_ts))
            minus_ub, rep_ts = heapq.heappop(pq)
        else:
            max_gain = -1
            visited = []

            while pq:
                minus_ub, cand_ts = heapq.heappop(pq)
                ub = len(list_of_simsets[cand_ts].intersection(set_of_uncovered_ts))  # marginal coverage of candidate time series
                visited.append((ub, cand_ts))
                
                if ub > max_gain:
                    max_gain = ub
                    rep_ts = cand_ts
                if pq and max_gain >= -pq[0][0]:
                    break

            if set_of_uncovered_ts:
                for ub, cand_ts in visited:
                    if cand_ts != rep_ts:
                        heapq.heappush(pq, (-ub, cand_ts))
        
        set_of_rep_ts.add(rep_ts)  # add representative time series with max marginal coverage
        set_of_uncovered_ts.difference_update(list_of_simsets[rep_ts])  # remove covered time series
        
    return set_of_rep_ts


def greedy(list_of_ts, tau, total_num_ts, num_ts_to_cover):
    """ select representative time series using Greedy """

    # populate with time series indices
    set_of_all_ts = set(np.arange(total_num_ts))
    set_of_uncovered_ts = set(np.arange(total_num_ts))
    set_of_rep_ts = set()

    # select representative time series
    while total_num_ts - len(set_of_uncovered_ts) < num_ts_to_cover:
        max_gain = 0
        rep_ts = None
        rep_simset = set()
        
        for cand_ts in set_of_all_ts - set_of_rep_ts:
            dist_matrix = pairwise_distances([list_of_ts[cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between candidate and all time series
            cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= tau}  # compute the similar set of candidate time series
                
            cur_gain = len(cand_simset.intersection(set_of_uncovered_ts))
            if cur_gain > max_gain:
                max_gain = cur_gain
                rep_ts = cand_ts
                rep_simset = cand_simset
        
        set_of_rep_ts.add(rep_ts)  # add representative time series with max marginal coverage
        set_of_uncovered_ts.difference_update(rep_simset)  # remove covered time series

    return set_of_rep_ts


def greedyET(list_of_ts, tau, total_num_ts, num_ts_to_cover):
    """ select representative time series using Greedy with Early Termination """

    # populate with time series indices
    set_of_all_ts = set(np.arange(total_num_ts))
    set_of_uncovered_ts = set(np.arange(total_num_ts))
    set_of_rep_ts = set()
    
    pq = []

    # select representative time series
    while total_num_ts - len(set_of_uncovered_ts) < num_ts_to_cover:
        rep_ts = None
        rep_simset = set()

        if not set_of_rep_ts:
            # initialize queue
            for cand_ts in set_of_all_ts:                
                dist_matrix = pairwise_distances([list_of_ts[cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between candidate and all time series
                ub = sum(1 for dist in dist_matrix[0] if dist <= tau)  # compute marginal coverage of candidate time series
                heapq.heappush(pq, (-ub, cand_ts))
            minus_ub, rep_ts = heapq.heappop(pq)

            dist_matrix = pairwise_distances([list_of_ts[rep_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between selected representative and all time series
            rep_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= tau}  # compute the similar set of selected representative time series
        else:
            max_gain = -1
            visited = []

            while pq:
                minus_ub, cand_ts = heapq.heappop(pq)
                dist_matrix = pairwise_distances([list_of_ts[cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between candidate and all time series
                cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= tau}  # compute the similar set of candidate time series
                ub = len(cand_simset.intersection(set_of_uncovered_ts))  # marginal coverage of candidate time series
                visited.append((ub, cand_ts))
                
                if ub > max_gain:
                    max_gain = ub
                    rep_ts = cand_ts
                    rep_simset = cand_simset
                if pq and max_gain >= -pq[0][0]:
                    break

            if set_of_uncovered_ts:
                for ub, cand_ts in visited:
                    if cand_ts != rep_ts:
                        heapq.heappush(pq, (-ub, cand_ts))
        
        set_of_rep_ts.add(rep_ts)  # add representative time series with max marginal coverage
        set_of_uncovered_ts.difference_update(rep_simset)  # remove covered time series

    return set_of_rep_ts


if __name__ == "__main__":
    args = parse_args()

    if args.method == "pregreedy" or args.method == "pregreedyET":
        list_of_simsets = load_simset(args)

        TOTAL_NUM_TS = len(list_of_simsets)
        NUM_TS_TO_COVER = round(TOTAL_NUM_TS * args.beta)
        
        t_start = time.time()

        if args.method == "pregreedy":
            set_of_rep_ts = pregreedy(list_of_simsets, TOTAL_NUM_TS, NUM_TS_TO_COVER)
        elif args.method == "pregreedyET":
            set_of_rep_ts = pregreedyET(list_of_simsets, TOTAL_NUM_TS, NUM_TS_TO_COVER)
    else:
        list_of_ts = load_data(args)

        # read max pairwise distance
        max_dist_dict = read_max_dist()
        if args.dataset not in max_dist_dict:
            raise Exception("Max distance not found. Run compute_dist.py first.")
        max_dist = max_dist_dict[args.dataset]
        
        TAU = max_dist * args.tau
        TOTAL_NUM_TS = len(list_of_ts)
        NUM_TS_TO_COVER = round(TOTAL_NUM_TS * args.beta)
        
        t_start = time.time()

        if args.method == "greedy":
            set_of_rep_ts = greedy(list_of_ts, TAU, TOTAL_NUM_TS, NUM_TS_TO_COVER)
        elif args.method == "greedyET":
            set_of_rep_ts = greedyET(list_of_ts, TAU, TOTAL_NUM_TS, NUM_TS_TO_COVER)
    
    print("| Time used: {:.3f} s".format(time.time() - t_start))
    print("| # of representative time series: {}".format(len(set_of_rep_ts)))
