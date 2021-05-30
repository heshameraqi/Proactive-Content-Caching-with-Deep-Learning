import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from data import MovieLensData
import math
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel, NCFModel  # Import Collaborative Filtering model architecture
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import pandas as pd
from math import floor
# Use GPU for Kerasimport tensorflow as tf
import keras
import tensorflow as tf
# config = tf.ConfigProto(device_count={'GPU':1, 'CPU':56})
# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)


class SimModel:
    def __init__(self, K_factor):
        self.K_factor = K_factor # The number of dimensional embeddings for movies and users in the CF deep learning model
        self.data = MovieLensData()
        # Load data and print statistics
        self.data.print_statistics()
        # self.data.data_movie_gap()
        # Shuffle data
        # shuffled_ratings = self.data.ratings.sample(frac=1., random_state=50)
        # self.users = shuffled_ratings['user_emb_id'].values
        # self.movies = shuffled_ratings['movie_emb_id'].values
        # self.ratings = shuffled_ratings['rating'].values
        self.users = self.data.ratings['user_emb_id'].values
        self.movies = self.data.ratings['movie_emb_id'].values
        # self.movies = [1,2,1,2,1,2,1,2]
        # self.movies = [1,2,1,2,3,4,3,4]
        self.ratings = self.data.ratings['rating'].values
        # print("self.data.max_userid: ", self.data.max_userid, " / self.data.max_movieid: ", self.data.max_movieid)

    def train_ncf_model(self, nb_step, cf_flag=True):
        # the default model will be ncf, cf will be used instead only if specified
        len_step = floor(len(self.movies)/nb_step)
        # we cut the ratings file into multiple intervals where the model will be trained in one interval and used in the next one
        for i in range(0, nb_step):
            inf_index = len_step * i
            # inf_index = 0  # in case we want to create intervals by going back each time to the beginning of the file
            if i == (nb_step-1):
                sup_index = len(self.movies)  # in case some ratings were left when the file was divided into multiple intervals
            else:
                sup_index = len_step * (i+1)
            par_users = self.users[inf_index:sup_index]
            par_movies = self.movies[inf_index:sup_index]
            par_ratings = self.ratings[inf_index:sup_index]
            if cf_flag:
                model = CFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
            else:
                model_ncf = NCFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
                model = model_ncf.model_final
            model.compile(loss='mse', optimizer='adamax')
            # Callbacks monitor the validation loss, save the model weights each time the validation loss has improved
            callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint(f'weights{i+1}.h5', save_best_only=True)]
            # Train the model: Use 30 epochs, 90% training data, 10% validation data
            history = model.fit([par_users, par_movies], par_ratings, nb_epoch=30, validation_split=.1, shuffle=True, batch_size=500, verbose=2, callbacks=callbacks)
            # Plot training and validation RMSE
            # loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ], 'training': [ math.sqrt(loss) for loss in history.history['loss'] ], 'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]})
            # ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
            # ax.set_ylabel("root mean squared error")
            # ax.set_ylim([0.0,3.0]);
            # Show the best validation RMSE
            min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
            print('Minimum RMSE at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

    @staticmethod
    def predict_rating(trained_model, user_id, movie_id):
        return trained_model.rate(user_id - 1, movie_id - 1)

    def apply_ncf_model(self, weights_file, cf_flag=False):
        # the default model will be ncf, cf will be used instead only if specified
        if cf_flag:
            trained_model = CFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
            # Load weights
            trained_model.load_weights(weights_file)
        else:
            trained_model = NCFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
            trained_model.model_final.load_weights(weights_file)
        rec_movies_list_all = list()
        # Use the pre-trained model
        for i in range(1, len(self.users)+1):
        # TODO: check which users to use (e.g., if a movie is recommended for more than X (threshold) users, it will be cached)
        # TODO: this will depend on the caching scheme
        # for i in range(1, 2):
            # Predict user ratings (enter user and his recommended movies --> get rating)
            # print("data.ratings: \n", data.ratings)
            user_ratings = self.data.ratings[self.data.ratings['user_id'] == i][['user_id', 'movie_id', 'rating']]
            user_ratings['prediction'] = user_ratings.apply(lambda x: SimModel.predict_rating(trained_model, i, x['movie_id']), axis=1)
            user_ratings = user_ratings.sort_values(by='rating', ascending=False).merge(self.data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
            # print(user_ratings)
            # Recommend user items (enter user and all movies --> get rating and sort them by best)
            # Remove from data.ratings the movies already rated/requested by the user and predict from the list of movies not yet rated
            recommendations = self.data.ratings[self.data.ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
            recommendations['prediction'] = recommendations.apply(lambda x: SimModel.predict_rating(trained_model, i, x['movie_id']), axis=1)
            recommendations = recommendations.sort_values(by='prediction', ascending=False).merge(self.data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
            # print(recommendations)
            rec_movies_list_user = recommendations["movie_id"].tolist()
            rec_movies_list_all.append(rec_movies_list_user)
            # print(recommended_movies_list)
        return rec_movies_list_all


if __name__ == "__main__":
    # Configurations
    # test_ratio = 0.1
    # batch_size = 256
    # split_mode = 'seq-aware'  # seq-aware or random
    # feedback = 'explicit'  # explicit or implicit
    K_FACTORS = 100
    nb_step = 2
    # cache_size = 10
    sim_model = SimModel(K_FACTORS)
    sim_model.train_ncf_model(nb_step)
    for i in range(0, nb_step):
        rec_movies_list_all = sim_model.apply_ncf_model(f'weights{i + 1}.h5')  # contains the recommended movies for each user
        flat_rec_list = list(dict.fromkeys([item for sublist in rec_movies_list_all for item in sublist]))  # flat list of recommended movies
        # print(rec_movies_list_all)



