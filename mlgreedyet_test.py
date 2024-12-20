import numpy as np
import time
import json
import os
import argparse
import heapq
from sklearn.metrics import pairwise_distances
import torch
from mlp import *
import pickle as pkl


SEED = 42
MAX_DIST_FILE = "max_dist.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="time series dataset")    
    parser.add_argument("--tau", type=float, required=True, help="normalized distance threshold between [0, 1]")
    parser.add_argument("--beta", type=float, default=1, help="coverage threshold between [0, 1]")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def load_data(args):
    """ load time series data """

    list_of_ts = np.load("data/{}.npy".format(args.dataset))
    return list_of_ts


def read_max_dist():
    """ read max pairwise distances from json file """

    if os.path.exists(MAX_DIST_FILE):
        with open(MAX_DIST_FILE, "r") as file:
            max_dist_dict = json.load(file)
        return max_dist_dict
    else:
        return {}


def load_embeddings(args):
    """ load time series embeddings """

    list_of_ts_emb = np.load("emb/{}.npy".format(args.dataset))
    return list_of_ts_emb


def aggregate(list_of_rep_ts, list_of_ts_emb):
    """ aggregate the time series embeddings based on the specified time series indices """

    if len(list_of_rep_ts):
        agg = np.sum(list_of_ts_emb[list_of_rep_ts], axis=0)
    else:
        len_of_emb = len(list_of_ts_emb[0])
        agg = np.zeros(len_of_emb)
    return agg
    

def load_reg(args):
    """ load the trained model """

    mlp_reg = MLP_REG().to(device)
    mlp_reg.load_state_dict(torch.load("models/{}".format(args.dataset)))
    mlp_reg.eval()
    scaler = pkl.load(open("models/{}_scaler.pkl".format(args.dataset), "rb"))
    return mlp_reg, scaler


def mlgreedyET(args, list_of_ts, list_of_ts_emb, max_dist, mlp_reg, scaler):
    """ use ML to predict TS marginal gain + select representative TS using GreedyET """

    NORM_TAU = args.tau
    TAU = NORM_TAU * max_dist
 
    TOTAL_NUM_TS = len(list_of_ts)
    NUM_TS_TO_COVER = round(TOTAL_NUM_TS * args.beta)

    list_of_ts_id = np.arange(TOTAL_NUM_TS)
    set_of_cand_ts = set(list_of_ts_id)
    set_of_uncovered_ts = set(list_of_ts_id)
    set_of_rep_ts = set()
    
    agg_of_all_ts = aggregate(list_of_ts_id, list_of_ts_emb)
    agg_cur_sol_emb = np.zeros(len(list_of_ts_emb[0]))
    
    pq = []

    while TOTAL_NUM_TS - len(set_of_uncovered_ts) < NUM_TS_TO_COVER:
        max_pred_gain = -np.inf
        rep_ts = None

        len_of_cur_sol = len(set_of_rep_ts)
        len_of_cur_res = TOTAL_NUM_TS - len_of_cur_sol

        if not set_of_rep_ts:
            # initialize the queue
            for cand_ts in set_of_cand_ts:
                cand_ts_emb = list_of_ts_emb[cand_ts]
                agg_cur_res_emb = np.subtract(np.subtract(agg_of_all_ts, agg_cur_sol_emb), cand_ts_emb)

                x_pred = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
                x_pred_scaled = scaler.transform(np.array([x_pred])).flatten()
                x_pred_scaled = torch.tensor(x_pred_scaled, dtype=torch.float, device=device)
                with torch.no_grad():
                    y_pred = mlp_reg(x_pred_scaled)  # predict marginal coverage
                ub = y_pred.data.cpu().numpy()[0]
                heapq.heappush(pq, (-ub, cand_ts))

            minus_ub, rep_ts = heapq.heappop(pq)
            max_pred_gain = -minus_ub
        else:
            visited = []
            while pq:
                minus_ub, cand_ts = heapq.heappop(pq)
                
                cand_ts_emb = list_of_ts_emb[cand_ts]
                agg_cur_res_emb = np.subtract(np.subtract(agg_of_all_ts, agg_cur_sol_emb), cand_ts_emb)

                x_pred = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
                x_pred_scaled = scaler.transform(np.array([x_pred])).flatten()
                x_pred_scaled = torch.tensor(x_pred_scaled, dtype=torch.float, device=device)
                with torch.no_grad():
                    y_pred = mlp_reg(x_pred_scaled)  # predict marginal coverage
                ub = y_pred.data.cpu().numpy()[0]
                visited.append((ub, cand_ts))

                if ub > max_pred_gain:
                    max_pred_gain = ub
                    rep_ts = cand_ts
                if pq and max_pred_gain >= -pq[0][0]:
                    break

            if set_of_uncovered_ts:
                for ub, cand_ts in visited:
                    if cand_ts != rep_ts:
                        heapq.heappush(pq, (-ub, cand_ts))

        dist_matrix = pairwise_distances([list_of_ts[rep_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between selected representative and all time series
        rep_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= TAU}  # compute the similar set of selected representative time series
        
        if not rep_simset.isdisjoint(set_of_uncovered_ts):  # if marginal coverage is > 0
            set_of_uncovered_ts.difference_update(rep_simset) # remove covered TS
            set_of_rep_ts.add(rep_ts)
            agg_cur_sol_emb += list_of_ts_emb[rep_ts]
        set_of_cand_ts.discard(rep_ts)        

    return set_of_rep_ts


def mlgreedy(args, list_of_ts, list_of_ts_emb, max_dist, mlp_reg, scaler):
    """ use ML to predict TS marginal gain + select representative TS using Greedy """

    NORM_TAU = args.tau
    TAU = NORM_TAU * max_dist
 
    TOTAL_NUM_TS = len(list_of_ts)
    NUM_TS_TO_COVER = round(TOTAL_NUM_TS * args.beta)

    list_of_ts_id = np.arange(TOTAL_NUM_TS)
    set_of_cand_ts = set(list_of_ts_id)
    set_of_uncovered_ts = set(list_of_ts_id)
    set_of_rep_ts = set()
    
    agg_of_all_ts = aggregate(list_of_ts_id, list_of_ts_emb)
    agg_cur_sol_emb = np.zeros(len(list_of_ts_emb[0]))

    while TOTAL_NUM_TS - len(set_of_uncovered_ts) < NUM_TS_TO_COVER:
        max_pred_gain = -np.inf
        rep_ts = None

        len_of_cur_sol = len(set_of_rep_ts)
        len_of_cur_res = TOTAL_NUM_TS - len_of_cur_sol

        for cand_ts in set_of_cand_ts:
            cand_ts_emb = list_of_ts_emb[cand_ts]
            agg_cur_res_emb = np.subtract(np.subtract(agg_of_all_ts, agg_cur_sol_emb), cand_ts_emb)

            x_pred = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
            x_pred_scaled = scaler.transform(np.array([x_pred])).flatten()
            x_pred_scaled = torch.tensor(x_pred_scaled, dtype=torch.float, device=device)
            with torch.no_grad():
                y_pred = mlp_reg(x_pred_scaled) # predict marginal coverage
            cur_pred_gain = y_pred.data.cpu().numpy()[0]
            
            if cur_pred_gain > max_pred_gain:
                max_pred_gain = cur_pred_gain
                rep_ts = cand_ts
        
        dist_matrix = pairwise_distances([list_of_ts[rep_ts]], list_of_ts, metric="euclidean", n_jobs=-1)  # compute pairwise distance between selected representative and all time series
        rep_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= TAU}  # compute the similar set of selected representative time series

        if not rep_simset.isdisjoint(set_of_uncovered_ts):  # if marginal coverage is > 0
            set_of_uncovered_ts.difference_update(rep_simset) # remove covered TS
            set_of_rep_ts.add(rep_ts)
            agg_cur_sol_emb += list_of_ts_emb[rep_ts]
        set_of_cand_ts.discard(rep_ts)        

    return set_of_rep_ts


if __name__ == "__main__":
    args = parse_args()
    set_seed(SEED)

    # read max pairwise distance
    max_dist_dict = read_max_dist()
    if args.dataset not in max_dist_dict:
        raise Exception("Max distance not found. Run compute_dist.py first.")
    max_dist = max_dist_dict[args.dataset]

    list_of_ts = load_data(args)
    list_of_ts_emb = load_embeddings(args)

    mlp_reg, scaler = load_reg(args)
    
    t_start = time.time()
    set_of_rep_ts = mlgreedyET(args, list_of_ts, list_of_ts_emb, max_dist, mlp_reg, scaler)

    print("| Time used: {:.3f} s".format(time.time() - t_start))
    print("| # of representative time series: {}".format(len(set_of_rep_ts)))