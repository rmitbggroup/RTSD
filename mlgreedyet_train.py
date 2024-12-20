import numpy as np
import time
import json
import os
import argparse
import heapq
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from mlp import *
import pickle as pkl


SEED = 42
MAX_DIST_FILE = "max_dist.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="time series dataset")    
    parser.add_argument("--tau", type=float, required=True, help="training normalized distance threshold between [0, 1]")
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
    

class DataLoader(object):
    def __init__(self, x, y, batch_size=256, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_idx = 0
        self.data_size = x.shape[0]
        if self.shuffle:
            self.reset()
    
    def reset(self):
        self.x, self.y = shuffle(self.x, self.y, random_state=SEED)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start_idx >= self.data_size:
            if self.shuffle:
                self.reset()
            self.start_idx = 0
            raise StopIteration
    
        batch_x = self.x[self.start_idx:self.start_idx+self.batch_size]
        batch_y = self.y[self.start_idx:self.start_idx+self.batch_size]

        batch_x = torch.tensor(batch_x, dtype=torch.float, device=device)
        batch_y = torch.tensor(batch_y, dtype=torch.float, device=device)

        self.start_idx += self.batch_size
        return (batch_x, batch_y)
    

def get_training_data(args, list_of_ts, list_of_ts_emb, max_dist):
    """ generate training data to train regressor for TS selection """

    X = []
    Y = []
    
    TOTAL_NUM_TS = len(list_of_ts)
    list_of_ts_id = np.arange(TOTAL_NUM_TS)

    t_start = time.time()

    set_of_all_ts = set(list_of_ts_id)
    set_of_uncovered_ts = set(list_of_ts_id)
    set_of_rep_ts = set()
    
    agg_all_ts_emb = aggregate(list_of_ts_id, list_of_ts_emb)
    agg_cur_sol_emb = np.zeros(len(list_of_ts_emb[0]))

    pq = []
    offsets = [0.1, -0.1]

    NORM_TAU = args.tau
    TAU = NORM_TAU * max_dist

    while set_of_uncovered_ts:
        max_gain = -np.inf
        rep_ts = None
        rep_simset = set()

        # prepare features
        len_of_cur_sol = len(set_of_rep_ts)
        len_of_cur_res = TOTAL_NUM_TS - len_of_cur_sol
        
        if not set_of_rep_ts:
            # initialize queue
            for cand_ts in set_of_all_ts:
                dist_matrix = pairwise_distances([list_of_ts[cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)
                cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= TAU}
                ub = len(cand_simset)  # marginal coverage of candidate time series
                heapq.heappush(pq, (-ub, cand_ts))

                # add to training data
                cand_ts_emb = list_of_ts_emb[cand_ts]
                agg_cur_res_emb = np.subtract(np.subtract(agg_all_ts_emb, agg_cur_sol_emb), cand_ts_emb)
                features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
                X.append(features)
                Y.append(np.array([ub]))
                
                # add contrastive training samples with different tau
                for offset in offsets:
                    cont_norm_tau = NORM_TAU + offset
                    cont_tau = cont_norm_tau * max_dist
                    cont_cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= cont_tau}  # compute similar set of candidate time series with different tau
                    
                    cont_features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, cont_norm_tau), axis=None)
                    cont_gain = len(cont_cand_simset.intersection(set_of_uncovered_ts))
                    X.append(cont_features)
                    Y.append(np.array([cont_gain]))
            
            _minus_ub, rep_ts = heapq.heappop(pq)
            rep_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= TAU}
        else:
            list_of_visited_ts = []
            set_of_visited_ts  = set()

            while pq:
                _minus_ub, cand_ts = heapq.heappop(pq)

                dist_matrix = pairwise_distances([list_of_ts[cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)
                cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= TAU}
                ub = len(cand_simset.intersection(set_of_uncovered_ts))  # marginal coverage of candidate time series

                list_of_visited_ts.append((ub, cand_ts))
                set_of_visited_ts.add(cand_ts)

                # add to training data
                cand_ts_emb = list_of_ts_emb[cand_ts]
                agg_cur_res_emb = np.subtract(np.subtract(agg_all_ts_emb, agg_cur_sol_emb), cand_ts_emb)
                features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
                X.append(features)
                Y.append(np.array([ub]))
                
                # add contrastive training samples with different tau
                for offset in offsets:
                    cont_norm_tau = NORM_TAU + offset
                    cont_tau = cont_norm_tau * max_dist
                    cont_cand_simset = {ts for ts, dist in enumerate(dist_matrix[0]) if dist <= cont_tau}
                    
                    cont_features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, cand_ts_emb, len_of_cur_sol, len_of_cur_res, cont_norm_tau), axis=None)
                    cont_gain = len(cont_cand_simset.intersection(set_of_uncovered_ts))  # marginal coverage of candidate time series with different tau
                    X.append(cont_features)
                    Y.append(np.array([cont_gain]))
                
                if ub > max_gain:
                    max_gain = ub
                    rep_ts = cand_ts
                    rep_simset = cand_simset
                
                # early termination
                if pq and max_gain >= -pq[0][0]:
                    # sample additional candidates from unvisited/pruned time series using kmeans++
                    list_of_unvis_ts = np.array(list(set_of_all_ts - set_of_rep_ts - set_of_visited_ts))
                    num_cand = round(0.1 * len(list_of_unvis_ts))
                    
                    _centers, indices = kmeans_plusplus(list_of_ts[list_of_unvis_ts], n_clusters=num_cand)
                    list_of_unvis_cand_ts = list_of_unvis_ts[indices]
                    
                    for unvis_cand_ts in list_of_unvis_cand_ts:
                        unvis_cand_ts_emb = list_of_ts_emb[unvis_cand_ts]
                        agg_cur_res_emb = np.subtract(np.subtract(agg_all_ts_emb, agg_cur_sol_emb), unvis_cand_ts_emb)
                        unvis_dist_matrix = pairwise_distances([list_of_ts[unvis_cand_ts]], list_of_ts, metric="euclidean", n_jobs=-1)
                        unvis_cand_simset = {ts for ts, dist in enumerate(unvis_dist_matrix[0]) if dist <= TAU}

                        # add to training data
                        unvis_features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, unvis_cand_ts_emb, len_of_cur_sol, len_of_cur_res, NORM_TAU), axis=None)
                        unvis_gain = len(unvis_cand_simset.intersection(set_of_uncovered_ts))  # marginal coverage of unvisited candidate time series
                        X.append(unvis_features)
                        Y.append(np.array([unvis_gain]))

                        # add contrastive training samples with different tau
                        for offset in offsets:
                            cont_norm_tau = NORM_TAU + offset
                            cont_tau = cont_norm_tau * max_dist
                            unvis_cont_cand_simset = {ts for ts, dist in enumerate(unvis_dist_matrix[0]) if dist <= cont_tau} 
                            
                            unvis_cont_features = np.concatenate((agg_cur_sol_emb, agg_cur_res_emb, unvis_cand_ts_emb, len_of_cur_sol, len_of_cur_res, cont_norm_tau), axis=None)
                            unvis_cont_gain = len(unvis_cont_cand_simset.intersection(set_of_uncovered_ts))  # marginal coverage of unvisited candidate time series with different tau
                            X.append(unvis_cont_features)
                            Y.append(np.array([unvis_cont_gain]))
                    break

            if set_of_uncovered_ts:
                for ub, cand_ts in list_of_visited_ts:
                    if cand_ts != rep_ts:
                        heapq.heappush(pq, (-ub, cand_ts))
        
        set_of_rep_ts.add(rep_ts)  # add representative time series with max marginal coverage
        agg_cur_sol_emb += list_of_ts_emb[rep_ts]
        set_of_uncovered_ts.difference_update(rep_simset)  # remove covered time series

    print("| Time used for generating data: {:.3f} s".format(time.time() - t_start))
    return np.array(X), np.array(Y)


def train_model(train_dataloader, valid_dataloader, model_name):
    """ train the model """

    mlp_reg = MLP_REG().to(device)
    optimizer = optim.Adam(mlp_reg.parameters())
    criterion = nn.L1Loss()
    
    EPOCHS = 300
    t_start = time.time()

    for epoch in range(EPOCHS):
        mlp_reg.train()
        for batch_x, batch_y in train_dataloader:
            y_pred = mlp_reg(batch_x)
            loss = criterion(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mlp_reg.eval()
        valid_loss = 0
        best_loss = np.inf
        num_batch = valid_dataloader.data_size // valid_dataloader.batch_size + 1

        with torch.no_grad():
            for batch_x, batch_y in valid_dataloader:
                y_pred = mlp_reg(batch_x)
                loss = criterion(y_pred, batch_y)
                valid_loss += loss.item()

            valid_loss /= num_batch
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(mlp_reg.state_dict(), model_name)
        print("Epoch: {} | Valid Loss: {:.3f}".format(epoch+1, valid_loss))
    print("| Time used for training: {:.3f} s".format(time.time() - t_start))


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

    # prepare training data
    X, Y = get_training_data(args, list_of_ts, list_of_ts_emb, max_dist)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    pkl.dump(scaler, open("models/{}_scaler.pkl".format(args.dataset), "wb"))
    train_dataloader = DataLoader(X_train_scaled, Y_train)
    valid_dataloader = DataLoader(X_valid_scaled, Y_valid)

    train_model(train_dataloader, valid_dataloader, "models/{}".format(args.dataset))