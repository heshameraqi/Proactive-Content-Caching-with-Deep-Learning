import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from models.NCF.NCF import NCF
from models.NCF_Att.NCF_Att import NCF_Att
import evaluate
import data_utils
import config
from main import test_model
import pandas as pd
import math


def fuse_n_top_k_items(user_rec, cache_size):
    # Fuse N Top-K to get unified Top-K across all users
    unique, unique_counts = np.unique(user_rec, return_counts=True)
    unique_counts_sorted = unique_counts.argsort()[::-1][:cache_size]  # sorting in descending order
    topk_movies = unique[unique_counts_sorted]
    # topk_movies = np.unique(user_rec)[:cache_size]
    # print("Top-K movie across the whole dataset: ", topk_movies)
    return topk_movies


def run_model_over_all_user_items_ids(base_dir, user_num, item_num, factor_num, num_layers, dropout, test_loader,
                                      top_k, cache_size, user_info, item_info,
                                      GMF_weights_path, MLP_weights_path, NeuMF_weights_path):
    # Load weights
    print("Loading Weights.....")
    GMF_weights = torch.load(base_dir + GMF_weights_path)
    MLP_weights = torch.load(base_dir + MLP_weights_path)
    NeuMF_weights = torch.load(base_dir + NeuMF_weights_path)
    # Create Model
    print("Creating Model.....")
    model = NCF(user_num, item_num, factor_num, num_layers, dropout, config.model, GMF_weights, MLP_weights)
    model.cuda()
    model.eval()

    # Calculate the model parameters
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # Test model performance on testing dataset
    HR, NDCG = evaluate.metrics(model, test_loader, top_k)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    # Get Top-K item for each user:
    items = torch.arange(item_num)
    users = torch.arange(user_num)
    user_rec = np.zeros((user_num, cache_size))
    for user in users:
        user = user.repeat(item_num)
        user = user.cuda()
        items = items.cuda()
        predictions = model(user, items, user_info, item_info)
        _, indices = torch.topk(predictions, cache_size)
        recommends = torch.take(items, indices).cpu().numpy().tolist()
        user_rec[user[0].item()] = recommends
    return user_rec


def test_caching(cache_size, cashing_type, L=None):
    """
    cashing_type: {string} one of the following options: [per_user, common]
      per_user: means each user will has it's one top-k cached items
      common: means all the user will has shared/centeralized top-k cached items
      hybrid: means we will group the user to L groups
    """
    # Create configuration
    print("Reading Configuration.....")
    base_dir = "../"
    factor_num = 32
    num_layers = 3
    dropout = 0.0
    test_num_ng = 99
    top_k = 10
    cache_size = cache_size

    # Data Loader
    # ############################# Load DATASET ##########################
    if config.user_item_info:
        train_data, test_data, user_num, item_num, train_mat, user_info, item_info = data_utils.load_all(base_dir=base_dir)
    else:
        train_data, test_data, user_num, item_num, train_mat = data_utils.load_all(base_dir=base_dir)
        user_info = None
        item_info = None
    test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
    test_loader = data.DataLoader(test_dataset, batch_size=test_num_ng + 1, shuffle=False, num_workers=0)

    test_data_timed = pd.read_csv(base_dir + config.test_rating, sep='\t', header=None,
                                  names=['user', 'item', 'rating', 'time'],
                                  usecols=[0, 1, 2, 3],
                                  dtype={0: np.int32, 1: np.int32,
                                         2: np.int32, 3: np.int32})
    test_data_timed = test_data_timed.values
    if config.rating_th != 0:
        test_data_timed = test_data_timed[np.where(test_data_timed[:, 2] >= config.rating_th)]
        print("number of requests in the data = ", len(test_data_timed))
    else:
        print("No rate's filtering will occur")
    print("The shape of the testing data is: ", test_data_timed.shape)

    if config.window_split:
        total_hit_rate = []
        intervals_num = len(train_data)
        for i in range(intervals_num-1):  # skip the last one
            GMF_weights_path = config.GMF_model_path.split('.pth')[0] + str(i) + ".pth"
            MLP_weights_path = config.MLP_model_path
            NeuMF_weights_path = config.NeuMF_model_path
            user_rec = run_model_over_all_user_items_ids(base_dir, user_num, item_num, factor_num, num_layers, dropout,
                                                         test_loader, top_k, cache_size, user_info, item_info,
                                                         GMF_weights_path, MLP_weights_path, NeuMF_weights_path)

            # Get Top-K
            user_rec = user_rec[np.unique(np.array(train_data[i])[:, 0])]  # take only the users in the training data
            topk_movies = fuse_n_top_k_items(user_rec, cache_size)

            # Calculate Hit Rate
            hit_count = 0
            test_data_timed = train_data[-1][(i + 1) * config.window_split * 1000:]
            for j in range(len(test_data_timed)):
                if test_data_timed[j][1] in topk_movies:
                    hit_count += 1
            hit_rate = (hit_count / len(test_data_timed)) * 100
            total_hit_rate.append(hit_rate)
            print("Hit Rate (For cache_size = ", cache_size, ") = ", hit_rate, "%")
        print("Total Hit Rate (For cache_size = ", cache_size, ") = ", sum(total_hit_rate)/len(total_hit_rate), "%")
    else:
        GMF_weights_path = config.GMF_model_path
        MLP_weights_path = config.MLP_model_path
        NeuMF_weights_path = config.NeuMF_model_path
        user_rec = run_model_over_all_user_items_ids(base_dir, user_num, item_num, factor_num, num_layers, dropout,
                                                     test_loader, top_k, cache_size, user_info, item_info,
                                                     GMF_weights_path, MLP_weights_path, NeuMF_weights_path)

        # Fuse N Top-K to get unified Top-K across all users
        if cashing_type == "hybrid":
            topk_movies_per_group = np.zeros((L, cache_size))
            num_users_per_group = int(user_num / L)
            for i in range(L):
                start = i * num_users_per_group
                end = start + num_users_per_group
                topk_movies = fuse_n_top_k_items(user_rec[start:end], cache_size)
                topk_movies_per_group[i] = topk_movies
        else:
            topk_movies = fuse_n_top_k_items(user_rec, cache_size)

        # calculate Hit Rate
        hit_count = 0
        for i in range(test_data_timed.shape[0]):
            if cashing_type == "common":
                temp_topk_movies = topk_movies
            elif cashing_type == "per_user":
                temp_topk_movies = user_rec[test_data_timed[i, 0]]
            elif cashing_type == "hybrid":
                group_id = test_data_timed[i, 0] // num_users_per_group
                temp_topk_movies = topk_movies_per_group[group_id]
                # temp_topk_movies = user_rec[group_id]
            else:
                print("Not implemented !!!")
            if test_data_timed[i, 1] in temp_topk_movies:
                hit_count += 1
        print("Hit Rate (For cache_size = ", cache_size, ") = ", (hit_count / test_data_timed.shape[0]) * 100, "%")


if __name__ == "__main__":
    test_caching(cache_size=100, cashing_type="hybrid", L=1)
    test_caching(cache_size=200, cashing_type="hybrid", L=1)
    test_caching(cache_size=300, cashing_type="hybrid", L=1)
    test_caching(cache_size=400, cashing_type="hybrid", L=1)
    test_caching(cache_size=500, cashing_type="hybrid", L=1)

    """
    test_caching(cache_size=500, cashing_type="hybrid", L=3020)
    test_caching(cache_size=400, cashing_type="common")

    test_caching(cache_size=100, cashing_type="per_user")

    test_caching(cache_size=500, cashing_type="common")

    test_caching(cache_size=500, cashing_type="per_user")
    """
