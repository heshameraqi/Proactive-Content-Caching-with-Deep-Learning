"""The MovieLens (https://movielens.org/) 100K dataset is comprised of 100K ratings, ranging from 1 to 5 stars,
from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies.
Some simple demographic information such as age, gender, genres for the users and items are also available.
The MovieLens 1M datasets is 10x larger."""

# Requirements & Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MovieLensData:
    # read self.ratings, users, and movies
    def __init__(self):
        MOVIELENS_DIR = os.path.join(os.path.join(os.getcwd(), 'data'), 'ml-1m')  # Specify the data to use

        # Specify User's Age and Occupation Column
        AGES = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
        OCCUPATIONS = {0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                       4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                       7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                       12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                       17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

        # Specify the data folders
        USER_DATA_FILE = 'users.dat'
        MOVIE_DATA_FILE = 'movies.dat'
        # MOVIE_DATA_FILE = 'movies_new.dat'  # movies file without gap in ids
        RATING_DATA_FILE = 'ratings.dat'
        # RATING_DATA_FILE = 'ratings_new.dat'  # ratings file without gap in ids
        # Read the Ratings File
        self.ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), sep='::', engine='python', encoding='latin-1', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        # self.ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), sep=':', engine='python', encoding='latin-1', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        # self.ratings = self.ratings.sort_values(by='timestamp', ascending=True)  # sorting requests/ratings by timestamp
        # Process self.ratings dataframe for Keras Deep Learning model (Add colums: user_emb_id=user_id-1 & movie_emb_id=movie_id-1)
        self.ratings['user_emb_id'] = self.ratings['user_id'] - 1
        self.ratings['movie_emb_id'] = self.ratings['movie_id'] - 1

        # Read the Users File and describe age and occupation
        self.users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), sep='::', engine='python', encoding='latin-1', names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
        self.users['age_desc'] = self.users['age'].apply(lambda x: AGES[x])
        self.users['occ_desc'] = self.users['occupation'].apply(lambda x: OCCUPATIONS[x])

        # Read the Movies File
        self.movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), sep='::', engine='python', encoding='latin-1', names=['movie_id', 'title', 'genres'])
        # self.movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), sep=':', engine='python', encoding='latin-1', names=['movie_id', 'title', 'genres'])

        # Set max_userid & max_movieid to the maximum user_id & movie_id in the self.ratings
        self.max_userid = self.ratings['user_id'].drop_duplicates().max()
        self.max_movieid = self.ratings['movie_id'].drop_duplicates().max()

        """# Define csv files to save Keras data into
        USERS_CSV_FILE = 'users.csv'
        MOVIES_CSV_FILE = 'movies.csv'
        RATINGS_CSV_FILE = 'self.ratings.csv'
        
        # Save into self.ratings.csv
        self.ratings.to_csv(RATINGS_CSV_FILE, sep='\t', header=True, encoding='latin-1', columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
        users.to_csv(USERS_CSV_FILE, sep='\t', header=True, encoding='latin-1', columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])
        movies.to_csv(MOVIES_CSV_FILE,  sep='\t',  header=True,  columns=['movie_id', 'title', 'genres'])
        print(len(self.ratings), 'self.ratings loaded and saved to CSV for Keras')
        print(len(users), 'descriptions of', max_userid, 'users loaded and saved to CSV for Keras')
        print(len(movies), 'descriptions of', max_movieid, 'movies loaded and saved to CSV for Keras')
        
        # Read the CSV Files
        self.ratings = pd.read_csv('self.ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
        max_userid = self.ratings['user_id'].drop_duplicates().max()
        max_movieid = self.ratings['movie_id'].drop_duplicates().max()
        users = pd.read_csv('users.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
        movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])"""

    # To avoid the gap in movies_ids by creating a new file for movies.dat and ratings.dat
    def data_movie_gap(self):
        movies = self.movies.copy()
        movies_new_ids = list(range(1, (movies.shape[0])+1))  # Ids starting from one
        # movies_new_ids = list(range(movies.shape[0]))  # Ids starting from zero
        movies['movie_new_id'] = movies_new_ids
        movies.to_csv(os.path.join(os.getcwd(), 'data', 'ml-1m', 'movies_new.dat'), index=False, header=False, sep=':', columns=['movie_new_id', 'title', 'genres'])
        ratings = self.ratings.copy()
        # Here we add the movie_new_id by using the ones assigned in ratings
        ratings['movie_new_id'] = ratings['movie_id'].apply(lambda x: movies['movie_new_id'][movies[movies['movie_id'] == x].index[0]])
        ratings.to_csv(os.path.join(os.getcwd(), 'data', 'ml-1m', 'ratings_new.dat'), index=False, header=False, sep=':', columns=['user_id', 'movie_new_id', 'rating', 'timestamp'])
    
    def print_statistics(self):
        # Print first 5 samples
        print(self.ratings.head(5))

        # Check data sparsity (main challenge for recommender systems)
        num_users = self.ratings.user_id.unique().shape[0]
        num_movies = self.ratings.movie_id.unique().shape[0]
        print(f'number of users: {num_users}, number of movies: {num_movies}')
        print(f'number of ratings: {len(self.ratings)}, num_users*num_movies: {(num_users * num_movies)}')

        # Plot histogram of the ratings
        """plt.figure()
        plt.hist(self.ratings['rating'], bins=5, ec='black')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Distribution of Ratings in MovieLens 1M')"""
        sns.set_style('whitegrid')
        sns.set(font_scale=1.5)
        sns.distplot(self.ratings['rating'].fillna(self.ratings['rating'].median()))
        plt.show(block=False)

    # Reads the dataframe line by line and enumerates the index of users/items start from zero.
    # The function then returns lists of users, items, ratings and a dictionary/matrix that records the interactions.
    # We can specify the type of feedback to either explicit or implicit.
    @staticmethod
    def load_data(ratings_data, feedback='explicit'):
        users, items, scores = [], [], []
        num_users = ratings_data.user_id.unique().shape[0]
        num_movies = ratings_data.movie_id.unique().shape[0]
        inter = np.zeros((num_movies, num_users)) if feedback == 'explicit' else {}
        for line in ratings_data.itertuples():
            user_index, item_index = int(line[1] - 1), int(line[2] - 1)
            score = int(line[3]) if feedback == 'explicit' else 1
            users.append(user_index)
            items.append(item_index)
            scores.append(score)
            if feedback == 'implicit':
                inter.setdefault(user_index, []).append(item_index)
            else:
                inter[item_index, user_index] = score
        return users, items, scores, inter

    def split_load_data(self, split_mode='random', test_ratio=0.1, feedback='explicit'):
        """seq-aware mode leaves out the item that a user rated most recently for
        test, and users' historical interactions as training set.
        User historical interactions are sorted from oldest to newest based on timestamp."""
        if split_mode == 'seq-aware':
            num_users = self.ratings.user_id.unique().shape[0]
            train_items, test_items, train_list = {}, {}, []
            for line in self.ratings.itertuples():
                u, i, rating, time = line[1], line[2], line[3], line[4]
                train_items.setdefault(u, []).append((u, i, rating, time))
                if u not in test_items or test_items[u][-1] < time:
                    test_items[u] = (i, rating, time)
            for u in range(1, num_users + 1):
                train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
            test_data = [(key, *value) for key, value in test_items.items()]
            train_data = [item for item in train_list if item not in test_data]
            train_data = pd.DataFrame(train_data)
            test_data = pd.DataFrame(test_data)
        else:
            mask = [True if x == 1 else False for x in np.random.uniform(0, 1, (len(self.ratings))) < 1 - test_ratio]
            neg_mask = [not x for x in mask]
            train_data, test_data = self.ratings[mask], self.ratings[neg_mask]

        train_u, train_i, train_r, train_inter = self.load_data(train_data, feedback)
        test_u, test_i, test_r, test_inter = self.load_data(test_data, feedback)

        return (train_u, train_i, train_r, train_inter), (test_u, test_i, test_r, test_inter)



