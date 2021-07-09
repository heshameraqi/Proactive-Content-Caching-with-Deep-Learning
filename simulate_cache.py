import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from data import MovieLensData
import math
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel, NCFModel  # Import Collaborative Filtering model architecture
from math import floor
import numpy as np
import Cache_Structure


class SimModel:
    def __init__(self, K_factor):
        self.K_factor = K_factor # The number of dimensional embeddings for movies and users in the CF deep learning model
        self.data = MovieLensData()
        # Load data and print statistics
        self.data.print_statistics()
        # self.data.remove_movie_gap()  # TODO: uncomment this
        self.users = self.data.ratings['user_emb_id'].values
        self.movies = self.data.ratings['movie_emb_id'].values
        # self.movies = [1,2,1,2,1,2,1,2]
        # self.movies = [1,2,1,2,3,4,3,4]
        self.ratings = self.data.ratings['rating'].values
        # print("self.data.max_userid: ", self.data.max_userid, " / self.data.max_movieid: ", self.data.max_movieid)

    def train_ncf_model(self, nb_step, cf_flag=False):
        # the default model will be ncf, cf will be used instead only if specified
        len_step = floor(len(self.movies)/nb_step)
        # we cut the ratings file into multiple intervals where the model will be trained in one interval and used in the next one
        for i in range(0, nb_step):
            print("=============== Training stage %d ==============="%(i))
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
                # model = CFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS).model
                model_used = CFModel(max(par_users)+1, max(par_movies)+1, K_FACTORS)
            else:
                # model = NCFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS).model
                model_used = NCFModel(max(par_users)+1, max(par_movies)+1, K_FACTORS)
            model = model_used.model
            model.compile(loss='mse', optimizer='adamax')
            # Callbacks monitor the validation loss, save the model weights each time the validation loss has improved
            callbacks = [EarlyStopping('val_loss', patience=10), ModelCheckpoint(f'weights{i+1}.h5', save_best_only=True)]
            # Train the model: Use 30 epochs, 90% training data, 10% validation data
            inputs = np.transpose(np.vstack((par_users, par_movies)))
            # history = model.fit(inputs, par_ratings, epochs=30, validation_split=.1, shuffle=True, batch_size=500, verbose=2, callbacks=callbacks)
            history = model.fit([par_users, par_movies], par_ratings, epochs=60, validation_split=.1, shuffle=True, batch_size=500, verbose=2, callbacks=callbacks)
            # Plot training and validation RMSE
            # loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ], 'training': [ math.sqrt(loss) for loss in history.history['loss'] ], 'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]})
            # ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
            # ax.set_ylabel("root mean squared error")
            # ax.set_ylim([0.0,3.0]);
            # Show the best validation RMSE
            min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
            print('Minimum RMSE at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

    def apply_ncf_model(self, weights_file, sub_users, par_data_ratings, max_user_id, max_movie_id, cf_flag=False):
        # the default model will be ncf, cf will be used instead only if specified
        if cf_flag:
            # trained_model = CFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
            trained_model = CFModel(max_user_id, max_movie_id, K_FACTORS)
        else:
            # trained_model = NCFModel(self.data.max_userid, self.data.max_movieid, K_FACTORS)
            trained_model = NCFModel(max_user_id, max_movie_id, K_FACTORS)
        trained_model.model.load_weights(weights_file)
        rec_movies_list_all = list()
        # If a movie is recommended for more than X (threshold) users, it will be cached
        # For every user TODO: should be every user in the past stages only not future
        # TODO: Batch the data to the model
        # for i in range(1, len(self.users)+1):
        for i in sub_users:
            # Predict user i ratings (enter user and his recommended movies --> get rating)
            # print("data.ratings: \n", data.ratings)
            # user_ratings = self.data.ratings[self.data.ratings['user_id'] == i][['user_id', 'movie_id', 'rating']]
            # user_ratings['prediction'] = user_ratings.apply(lambda x: trained_model.rate(i, x['movie_id']), axis=1)
            # user_ratings = user_ratings.sort_values(by='rating', ascending=False).merge(self.data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
            # print(user_ratings)
            # Recommend user items (enter user and all movies --> get rating and sort them by best)
            # Remove from data.ratings the movies already rated/requested by the user and predict from the list of movies not yet rated
            user_ratings = par_data_ratings[par_data_ratings['user_id'] == i][['user_id', 'movie_id', 'rating']]
            recommendations = par_data_ratings[par_data_ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
            recommendations['prediction'] = recommendations.apply(lambda x: trained_model.rate(i, x['movie_id']), axis=1)
            recommendations = recommendations.sort_values(by='prediction', ascending=False).merge(self.data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
            # print(recommendations)
            rec_movies_list_user = recommendations["movie_id"].tolist()
            rec_movies_list_all.append(rec_movies_list_user)
            # print(recommended_movies_list)
        return rec_movies_list_all

    def sim_ncf_cache(self, network):
        hit_g, miss_g = 0, 0
        hit_cache, miss_cache, hit_ratio_caches_aux, hit_ratio_caches = ([0]*network.nb_nodes for _ in range(4))
        len_step = floor(len(self.movies)/nb_step)
        for i in range(0, nb_step):
            inf_index = len_step * i
            if i == (nb_step - 1):
                sup_index = len(self.movies)  # in case some ratings were left when the file was divided into multiple intervals
            else:
                sup_index = len_step * (i + 1)
            print("=============== Training stage %d [%d, %d] ===============" % (i+1, inf_index, sup_index))
            par_movies = self.movies[inf_index:sup_index]
            par_users = self.users[inf_index:sup_index]
            par_data_ratings = self.data.ratings[inf_index:sup_index]
            sub_users = self.data.ratings['user_id'][inf_index:sup_index].drop_duplicates().values
            flat_rec_list = list()
            if i == 0:
                hit_g_aux, miss_g_aux, hit_cache_aux, miss_cache_aux = network.hit_ratio_multi_lru(par_movies)
                hit_ratio_global_aux = float(hit_g_aux / (hit_g_aux + miss_g_aux))
                hit_ratio_avg_aux = float(sum(hit_cache_aux) / (sum(hit_cache_aux) + sum(miss_cache_aux)))
                for j in range(0, network.nb_nodes):
                    if hit_cache_aux[j] == 0:
                        hit_ratio_caches_aux[j] = 0
                    else:
                        hit_ratio_caches_aux[j] = float(hit_cache_aux[j] / (hit_cache_aux[j] + miss_cache_aux[j]))
                print(f"on step {i+1} - hit_ratio_global: ", hit_ratio_global_aux*100, " / hit_ragio_avg: ", hit_ratio_avg_aux*100, " / hit_ratio_caches: ", hit_ratio_caches_aux[0]*100)
                hit_g += hit_g_aux
                miss_g += miss_g_aux
                hit_cache = [x + y for x, y in zip(hit_cache, hit_cache_aux)]
                miss_cache = [x + y for x, y in zip(miss_cache, miss_cache_aux)]
                rec_movies_list_all = self.apply_ncf_model(f'weights{i+1}.h5', sub_users, par_data_ratings, max(par_users)+1, max(par_movies)+1) # 
                flat_rec_list = list(dict.fromkeys([item for sublist in rec_movies_list_all for item in sublist]))  # flat list of recommended movies
            else:
                sub_users = self.data.ratings['user_id'][inf_index:sup_index].drop_duplicates().values
                # rec_movies_list_all: contains the recommended movies for each user
                rec_movies_list_all = self.apply_ncf_model(f'weights{i}.h5', sub_users, max(sub_users)+1, max(par_movies)+1)
                flat_rec_list = list(dict.fromkeys([item for sublist in rec_movies_list_all for item in sublist]))  # flat list of recommended movies
                if i == 1:
                    prev_cache = network.get_cache()
                    network.cache_init(prev_cache)
                hit_g_aux, miss_g_aux, hit_cache_aux, miss_cache_aux = network.hit_ratio_multi_lru_cfl(par_movies, sub_users, flat_rec_list)
                # hit_g_aux, miss_g_aux, hit_cache_aux, miss_cache_aux = network.hit_ratio_multi_lru(par_movies)
                hit_ratio_global_aux = float(hit_g_aux / (hit_g_aux + miss_g_aux))
                hit_ratio_avg_aux = float(sum(hit_cache_aux) / (sum(hit_cache_aux) + sum(miss_cache_aux)))
                for j in range(0, network.nb_nodes):
                    if hit_cache_aux[j] == 0:
                        hit_ratio_caches_aux[j] = 0
                    else:
                        hit_ratio_caches_aux[j] = float(hit_cache_aux[j] / (hit_cache_aux[j] + miss_cache_aux[j]))
                print(f"on step {i + 1} - hit_ratio_global: ", hit_ratio_global_aux*100, " / hit_ragio_avg: ", hit_ratio_avg_aux*100, " / hit_ratio_caches: ", hit_ratio_caches_aux[0]*100)
                hit_g += hit_g_aux
                miss_g += miss_g_aux
                hit_cache = [x + y for x, y in zip(hit_cache, hit_cache_aux)]
                miss_cache = [x + y for x, y in zip(miss_cache, miss_cache_aux)]
                rec_movies_list_all = self.apply_ncf_model(f'weights{i+1}.h5', sub_users, par_data_ratings, max(par_users)+1, max(par_movies)+1)
                flat_rec_list = list(dict.fromkeys([item for sublist in rec_movies_list_all for item in sublist]))
        hit_ratio_global = float(hit_g / (hit_g + miss_g))
        hit_ratio_avg = float(sum(hit_cache) / (sum(hit_cache) + sum(miss_cache)))
        for j in range(0, network.nb_nodes):
            hit_ratio_caches[j] = float(hit_cache[j] / (hit_cache[j] + miss_cache[j]))
        return hit_ratio_global, hit_ratio_avg, hit_ratio_caches

if __name__ == "__main__":
    # Configurations
    # test_ratio = 0.1
    # batch_size = 256
    # split_mode = 'seq-aware'  # seq-aware or random
    # feedback = 'explicit'  # explicit or implicit
    K_FACTORS = 100
    nb_step = 4
    cache_size = 100
    sim_model = SimModel(K_FACTORS)
    sim_model.train_ncf_model(nb_step)
    '''for i in range(0, nb_step):
        print("=============== Testing stage %d ==============="%(i))
        rec_movies_list_all = sim_model.apply_ncf_model(f'weights{i + 1}.h5')  # contains the recommended movies for each user
        flat_rec_list = list(dict.fromkeys([item for sublist in rec_movies_list_all for item in sublist]))  # flat list of recommended movies
        # print(rec_movies_list_all)'''
    network = Cache_Structure.MultiCache(cache_size, 'network.txt', True)
    hit_ratio_global, hit_ratio_avg, hit_ratio_caches = sim_model.sim_ncf_cache(network)
    print(f"final results - hit_ratio_global: {hit_ratio_global*100}/ hit_ratio_avg: {hit_ratio_avg*100}/ hit_ratio_caches: {hit_ratio_caches[0]*100}")
