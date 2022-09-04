import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
import config
import random


def load_all(base_dir="./", test_num=100):
    """
    We load all the three file here to save time in each epoch.
    returns:
    train_data: Each Line is a training instance: userID itemID rating timestamp (if have)
    test_data: Each Line is a testing instance: userID itemID rating timestamp (if have)
    user_num: (n) number of users in dataset
    item_num: (m) number of items in dataset
    train_mat: (n,m) matrix where 1 is set to indicate if the user has ratted this item or not.
    """
    train_data = pd.read_csv(base_dir+config.train_rating, sep='\t', header=None,
                                   names=['user', 'item', 'rating', 'time'],
                                   usecols=[0, 1, 2, 3], dtype={0: np.int32, 1: np.int32, 2: np.int32, 3: np.int32})
    train_data = train_data.values
    user_num = train_data[:, 0].max() + 1
    item_num = train_data[:, 1].max() + 1
    train_data = train_data.tolist()

    if config.rating_th != 0:
        train_data_rating_sorting = sorted(train_data, key=lambda x: x[2], reverse=True)  # sort based on ratings
        train_data_rating_sorting = np.array(train_data_rating_sorting)
        x1, x2, x3 = np.unique(train_data_rating_sorting[:, 2], return_index=True, return_counts=True)
        print("unique rating values: ", x1)
        print("unique rating counts: ", x3)
        print("Starting Filtering rating based on thresholds...")
        train_data = train_data_rating_sorting[np.where(train_data_rating_sorting[:, 2] >= config.rating_th)]
        print("number of requests in the data = ", len(train_data))
        x1, x2, x3 = np.unique(train_data[:, 2], return_index=True, return_counts=True)
        print("unique rating values: ", x1)
        print("unique rating counts: ", x3)
    else:
        print("No rate's filtering will occur")

    if config.time_sorting:
        train_data_timed_sorting = sorted(train_data, key=lambda x: x[3], reverse=False)  # sort based on time
        train_data = np.array(train_data_timed_sorting)

    train_data = np.array(train_data)

    """
    train_data = pd.read_csv(config.train_rating,
                             sep='\t', header=None, names=['user', 'item'],
                             usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1
    train_data = train_data.values.tolist()
    """

    # Load ratings as a dok matrix, to create the implicit feedback.
    # matrix (n,m) where 1 is set to indicate if the user has ratted this item or not.
    # skip it in the end-to-end case as we will not generate negative samples.
    if config.E2E:
        train_mat = None
    else:
        train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
        for x in train_data:
            train_mat[x[0], x[1]] = 1.0

    intervals_num = math.ceil(len(train_data) / (config.window_split * 1000))
    if config.window_split:
        intervals = []
        if not config.time_sorting:
            # Distribute the data to make sure that all the users are represented in each time-interval
            print("Distribute the data to make sure that all the users are represented in each time-interval...")
            users_unique_ids, users_indexs, users_counts = np.unique(train_data[:, 0], return_index=True,
                                                                     return_counts=True)
            # min. interaction for a user is 19 & max. interactions for a user is 2313
            # list of lists. The first list for the whole users and for each user we have list of intervals
            idxForEachUser = []
            for uniqUserId in users_unique_ids:
                idxForEachUser.append(np.array_split(np.where(train_data[:, 0] == uniqUserId)[0], intervals_num))
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
            newTrainData = np.zeros_like(train_data)
            startIdx = 0
            for intervalIdx in range(intervals_num):  # loop on intervals
                endIdx = startIdx + len(intervalDataIdx[intervalIdx])
                newTrainData[startIdx:endIdx] = train_data[intervalDataIdx[intervalIdx]]
                startIdx = endIdx
            train_data = newTrainData

        for i in range(intervals_num):
            start_idx = 0
            end_idx = (i+1)*config.window_split*1000
            temp = np.stack((train_data[start_idx: end_idx, 0], train_data[start_idx: end_idx, 1]), axis=-1)
            intervals.append(temp.tolist())
            # get unique users & items:
            temp_user = np.unique(train_data[start_idx: end_idx, 0], return_counts=True)[0]
            temp_movie = np.unique(train_data[start_idx: end_idx, 1], return_counts=True)[0]
        train_data = intervals
    else:
        train_data = train_data.tolist()

    # Recommender test data:
    # ----------------------
    # first item in the test_data is user-item which is rated
    # and others are items that are unrated by this user.
    test_data = []
    with open(base_dir + config.test_negative, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()

    if config.E2E:
        # prepare new data:
        recommenderTestData = test_data
        newTrainData = []
        test_data_E2E = []
        for i in range(1, intervals_num-1):
            newTrainData.append(train_data[i])
            test_data_E2E.append(train_data[-1][len(train_data[i]):])
        train_data = newTrainData
        test_data = [recommenderTestData, test_data_E2E]

    # Read User and Item Informations:
    if config.user_item_info:
        # Read user info --> [UserID::Gender::Age::Occupation::Zip-code]
        user_info = pd.read_csv("Data/MovieLens_1M_ORG/users.dat", sep='::',
                                engine='python', header=None,
                                names=["UserID", "Gender", "Age", "Occupation", "Zipcode"])
        user_info.Gender = user_info.Gender.astype('category').cat.codes
        user_info.Age = user_info.Age.astype('category').cat.codes
        user_info.Zipcode = user_info.Zipcode.astype('category').cat.codes
        user_info = user_info.to_numpy()
        # Read item info --> [MovieID::Title::Genres]
        ratings = pd.read_csv("Data/MovieLens_1M_ORG/ratings.dat", sep='::',
                              engine='python', header=None,
                              names=["UserID", "MovieID", "Rating", "Timestamp"])
        item_info = pd.read_csv("Data/MovieLens_1M_ORG/movies.dat", sep='::',
                                engine='python', header=None, usecols=[0, 2],
                                names=["MovieID", "Genres"])
        df_merged = pd.merge(ratings, item_info, on='MovieID', how='inner')
        df_merged = df_merged[["MovieID", "Genres"]]
        df_merged.drop_duplicates(subset="MovieID", keep="first", inplace=True)
        df_merged = df_merged.sort_values(by=['MovieID'])
        df_merged = df_merged[["Genres"]]
        df_merged = pd.get_dummies(df_merged.Genres, prefix='Genres')
        item_info = df_merged.to_numpy()
    """
    train_data =  (994169, 2)
    test_data =  (604000, 2) [num_user*100, 2]
    user_num =  6040
    item_num =  3706
    train_mat =  (6040, 3706)
    """
    """
    # Save E2E testing data to be used with Optimal Library:
    f = open("e2e_trace_1.txt", "w+")
    uniq_counter = 1
    for item_pair in test_data_E2E[0]:
        f.write("%d %d %d\n" % (uniq_counter, item_pair[1], 1024))
        uniq_counter += 1
    """

    if config.user_item_info:
        return train_data, test_data, user_num, item_num, train_mat, user_info, item_info
    else:
        return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0,
                 is_training=None, user_info=None, item_info=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]
        self.user_info = user_info
        self.item_info = item_info
        if config.E2E and self.is_training:
            # prepare new data:
            temp = self.features_ps
            temp = np.array(temp)
            self.features_ps = temp[:-config.window_split*1000].tolist()
            # TODO: eslam try to calculate cache prob based on the second half only.
            self.all_features_ps = temp
            #self.all_features_ps = temp[-config.window_split*1000:]

    def ng_sample(self):
        """
        Generate negative samples for training stage only, with the help of "train_mat".
        For each interaction in the training we generate N negative samples for this interaction.
        """
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        if config.E2E:
            return len(self.features_ps)
        else:
            return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        if self.is_training and not config.E2E:
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        if config.E2E and self.is_training:
            cacheProb = np.zeros(self.num_item)
            uniValue, UniCount = np.unique(self.all_features_ps[self.all_features_ps[:, 0] == user, 1],
                                           return_counts=True)  # get top-k foe specific user
            cacheProb[uniValue] = UniCount
            label = cacheProb / cacheProb.max()  # convert it to prob

        if config.user_item_info:
            target = {'label': label, 'user_info': self.user_info[user], 'item_info': self.item_info[item]}
        else:
            target = {'label': label}
        return user, item, target
# if config.user_item_info:
#		return user, item, label, self.user_info[user], self.item_info[item]
# return user, item, label
