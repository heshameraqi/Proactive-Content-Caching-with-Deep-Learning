import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import matplotlib.pyplot as plt
import config
import random
import math


def fuse_n_top_k_items(user_rec, cache_size):
    # Fuse N Top-K to get unified Top-K across all users
    unique, unique_counts = np.unique(user_rec, return_counts=True)
    # Exclude the -1:
    unique = unique[1:]
    unique_counts = unique_counts[1:]
    unique_counts_sorted = unique_counts.argsort()[::-1][:cache_size]  # sorting in descending order
    topk_movies = unique[unique_counts_sorted]
    # topk_movies = np.unique(user_rec)[:cache_size]
    # print("Top-K movie across the whole dataset: ", topk_movies)
    return topk_movies


def calc_top_k(users, movies, L, cache_size):
    user_movies_matrix = np.zeros((num_of_users_in_dataset, num_of_movies_in_dataset)) - 1
    for j in range(len(users)):  # Create the sparse matrix
        user_movies_matrix[users[j], movies[j]] = movies[j]

    # Get Top-K for each group
    topk_movies_per_group = np.zeros((L, cache_size))
    num_users_per_group = int(num_of_users_in_dataset / L)
    for i in range(L):
        start = i * num_users_per_group
        end = start + num_users_per_group
        topk_movies = fuse_n_top_k_items(user_movies_matrix[start:end], cache_size)
        topk_movies_per_group[i] = topk_movies

    return topk_movies_per_group


def calculate_hit_rate(topk_movies_per_group, testing_data):
    hit_count = 0
    for i in range(testing_data.shape[0]):
        group_id = testing_data[i, 0] // num_users_per_group
        temp_topk_movies = topk_movies_per_group[group_id]
        # temp_topk_movies = user_rec[group_id]
        if testing_data[i, 1] in temp_topk_movies:
            hit_count += 1
    hite_rate = (hit_count / testing_data.shape[0]) * 100
    return hite_rate


def average(lst):
    return sum(lst) / len(lst)


if __name__ == "__main__":
    random.seed(2)
    # ----------------
    # Configuration:
    # ----------------
    base_path = "../"
    possible_ws_cache_size = [100]
    possible_ws = [config.window_split]
    Ls = [1, 10, 20, 40, 151]
    cache_sizes = [100, 200, 300, 400, 500]
    Ls = [1]
    use_future_data = False  # Enable it to use the whole data(even the future one)(No intervals if enabled)
    # --------------------------------------------------------------------------------------------------------

    # ----------------
    # Loading dataset:
    # ----------------
    train_data_timed = pd.read_csv(base_path+config.train_rating, sep='\t', header=None, names=['user', 'item',  'rating', 'time'],
                                   usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int32, 2: np.int32, 3: np.int32})

    train_data_timed = train_data_timed.values.tolist()
    train_data_timed_sorted = sorted(train_data_timed, key=lambda x: x[2], reverse=True)  # sort based on ratings
    train_data_timed_sorted = np.array(train_data_timed_sorted)
    x1, x2, x3 = np.unique(train_data_timed_sorted[:,2], return_index=True, return_counts=True)
    print("unique rating values: ", x1)
    print("unique rating counts: ", x3)
    print("Starting Filtering rating based on thresholds...")
    if config.rating_th != 0:
        train_data_timed_sorted = train_data_timed_sorted[np.where(train_data_timed_sorted[:, 2] >= config.rating_th)]
        print("number of requests in the data = ", len(train_data_timed_sorted))
        #idx = np.argmax(train_data_timed_sorted[:, 2] < config.rating_th)
        #print("number of requests in the data = ", idx)
        #train_data_timed_sorted = train_data_timed_sorted[:idx]
    else:
        print("No rate's filtering will occur")
    num_of_users_in_dataset = train_data_timed_sorted[:, 0].max() + 1
    num_of_movies_in_dataset = train_data_timed_sorted[:, 1].max() + 1

    x1, x2, x3 = np.unique(train_data_timed_sorted[:, 2], return_index=True, return_counts=True)
    print("unique rating values: ", x1)
    if config.time_sorting:
        train_data_timed_sorted = sorted(train_data_timed_sorted, key=lambda x: x[3], reverse=False)  # sort based on time
        train_data_timed_sorted = np.array(train_data_timed_sorted)
    else:
        intervals_num = math.ceil(len(train_data_timed_sorted) / (config.window_split * 1000))
        intervals = []
        # Distribute the data to make sure that all the users are represented in each time-interval
        print("Distribute the data to make sure that all the users are represented in each time-interval...")
        users_unique_ids, users_indexs, users_counts = np.unique(train_data_timed_sorted[:, 0], return_index=True,
                                                                 return_counts=True)
        # min. interaction for a user is 19 & max. interactions for a user is 2313
        # list of lists. The first list for the whole users and for each user we have list of intervals
        idxForEachUser = []
        for uniqUserId in users_unique_ids:
            idxForEachUser.append(np.array_split(np.where(train_data_timed_sorted[:, 0] == uniqUserId)[0], intervals_num))
        # list of arrays each one contains the indices from the data for this interval
        intervalDataIdx = []
        for intervalIdx in range(intervals_num):  # loop on intervals
            temp = []
            for uniqUserId in range(len(users_unique_ids)):  # loop on users
                temp.append(idxForEachUser[uniqUserId][intervalIdx])
            temp = np.hstack(temp)
            random.shuffle(temp)
            intervalDataIdx.append(temp)
        # prepare the train data
        newTrainData = np.zeros_like(train_data_timed_sorted)
        startIdx = 0
        for intervalIdx in range(intervals_num):  # loop on intervals
            endIdx = startIdx + len(intervalDataIdx[intervalIdx])
            newTrainData[startIdx:endIdx] = train_data_timed_sorted[intervalDataIdx[intervalIdx]]
            startIdx = endIdx
        train_data_timed_sorted = newTrainData

    print(train_data_timed_sorted.shape)
    # ---------------------------------------------------------------------------------------------------------

    # -----------------------------
    # Split data into window sizes:
    # -----------------------------
    possible_ws = [x * 1000 for x in possible_ws]
    number_of_interactions = train_data_timed_sorted.shape[0]  # 994169
    ws_users_dict = {}
    ws_movies_dict = {}
    for ws in possible_ws:
        print("--------------------------------------")
        print("      Processing Window Size = ", ws)
        print("--------------------------------------")
        unique_users = []
        unique_movies = []
        for i in range(int(np.ceil(number_of_interactions / ws))):
            start = i * ws
            end = (i + 1) * ws
            temp_user = train_data_timed_sorted[start:end, 0]
            temp_movie = train_data_timed_sorted[start:end, 1]
            unique_users.append(temp_user)
            unique_movies.append(temp_movie)
            print("Num of users in window # ", i, " is ", len(temp_user))
        ws_users_dict[ws] = unique_users
        ws_movies_dict[ws] = unique_movies
    # -----------------------------------------------------------------------------------------------------------

    # ----------------
    # Compute Hit-Rate:
    # ----------------
    for L in Ls:
        for cache_size in cache_sizes:
            if use_future_data:
                # Merge all items from different window_sizes as in this approach no need for window concept.
                # Training data --> all dataset
                # Testing data --> is the original NCF test-set.
                movies = np.concatenate(ws_movies_dict[ws], axis=0)  # (994169,)
                users = np.concatenate(ws_users_dict[ws], axis=0)  # (994169,)
                topk_movies_per_group = calc_top_k(users, movies, L, cache_size)
                # prepare testing data:
                num_users_per_group = int(num_of_users_in_dataset / L)
                test_data_timed = pd.read_csv(base_path+config.test_rating, sep='\t', header=None,
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
                # calculate hit rate:
                hit_rate = calculate_hit_rate(topk_movies_per_group, test_data_timed)
                print("Hit Rate (For cache_size = ", cache_size, " / For L = ", L, ") = ", hit_rate, "%")
            else:
                for ws in possible_ws:
                    # Training data --> current and previous intervals only.
                    # Testing data --> next intervals.
                    print("--------------------------------------")
                    print("      Processing Window Size = ", ws)
                    print("--------------------------------------")
                    total_hit_rate = []
                    unique_movies_intervals = ws_movies_dict[ws]
                    unique_users_intervals = ws_users_dict[ws]
                    for i in range(len(unique_movies_intervals) - 1):  # Loop on intervals based on selected window size
                        # Stack all items from current and preceding intervals
                        movies = np.hstack(np.array(unique_movies_intervals[:i + 1]))
                        users = np.hstack(np.array(unique_users_intervals[:i + 1]))
                        topk_movies_per_group = calc_top_k(users, movies, L, cache_size)
                        # prepare testing data:
                        num_users_per_group = int(num_of_users_in_dataset / L)
                        future_movies = np.hstack(np.array(unique_movies_intervals[i + 1:]))
                        future_users = np.hstack(np.array(unique_users_intervals[i + 1:]))
                        test_data = np.swapaxes(np.vstack([future_users, future_movies]), 0, 1)
                        # calculate hit rate:
                        hit_rate = calculate_hit_rate(topk_movies_per_group, test_data)
                        total_hit_rate.append(hit_rate)
                    print("Hit Rate (For cache_size = ", cache_size, " / For L = ", L, ") = ", average(total_hit_rate), "%")