from data import MovieLensData
import math
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel  # Import Collaborative Filtering model architecture
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

# Configurations
test_ratio = 0.1
batch_size = 256
split_mode = 'seq-aware'  # seq-aware or random
feedback = 'explicit'  # explicit or implicit
K_FACTORS = 100  # The number of dimensional embeddings for movies and users in the CF deep learning model

# Use GPU for Kerasimport tensorflow as tf
import keras
import tensorflow as tf
config = tf.ConfigProto(device_count={'GPU':1, 'CPU':56})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

# Load data and print statistics
data = MovieLensData()
data.print_statistics()

# Shuffle data
shuffled_ratings = data.ratings.sample(frac=1., random_state=50)
users = shuffled_ratings['user_emb_id'].values
movies = shuffled_ratings['movie_emb_id'].values
ratings = shuffled_ratings['rating'].values

# Define model
model = CFModel(data.max_userid, data.max_movieid, K_FACTORS)
# model_ncf = NCFModel(data.max_userid, data.max_movieid, K_FACTORS)
# model = model_ncf.model_final
model.compile(loss='mse', optimizer='adamax')

# Callbacks monitor the validation loss, save the model weights each time the validation loss has improved
callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('weights.h5', save_best_only=True)]

# Train the model: Use 30 epochs, 90% training data, 10% validation data
history = model.fit([users, movies], ratings, nb_epoch=30, validation_split=.1, verbose=2, callbacks=callbacks)

# Show the best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

# Use the pre-trained model
trained_model = CFModel(data.max_userid, data.max_movieid, K_FACTORS)
# trained_model_obj = NCFModel(data.max_userid, data.max_movieid, K_FACTORS)
# trained_model = trained_model_obj.model_final
# Load weights
trained_model.load_weights('weights.h5')

# Predict user ratings (enter user and his recommended movies --> get rating)
TEST_USER = 2000
data.users[data.users['user_id'] == TEST_USER]

def predict_rating(user_id, movie_id):
    return trained_model.rate(user_id - 1, movie_id - 1)
	# return trained_model_obj.rate(user_id - 1, movie_id - 1)

# user_ratings = data.ratings[ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
user_ratings = data.ratings[data.ratings['user_id'] == TEST_USER][['user_id', 'movie_id', 'rating']]
user_ratings['prediction'] = user_ratings.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
# user_ratings.sort_values(by='rating', ascending=False).merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
user_ratings = user_ratings.sort_values(by='rating', ascending=False).merge(data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
print(user_ratings)

# Recommend user items (enter user and all movies --> get rating and sort them by best)
recommendations = data.ratings[data.ratings['movie_id'].isin(user_ratings['movie_id']) == False][['movie_id']].drop_duplicates()
recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(TEST_USER, x['movie_id']), axis=1)
# recommendations.sort_values(by='prediction', ascending=False).merge(movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
recommendations = recommendations.sort_values(by='prediction', ascending=False).merge(data.movies, on='movie_id', how='inner', suffixes=['_u', '_m']).head(20)
print(recommendations)

# Afterwards, we put the above steps together and it will be used in the next section. The results are wrapped with Dataset and DataLoader.
# Note that the last_batch of DataLoader for training data is set to the rollover mode (The remaining samples are rolled over to the next epoch.) and orders are shuffled.
# (train_u, train_i, train_r, train_inter), (test_u, test_i, test_r, test_inter) = data.split_load_data(split_mode, test_ratio, feedback)
#train_set = data.ArrayDataset(np.array(train_u), np.array(train_i), np.array(train_r))
#test_set = data.ArrayDataset(np.array(test_u), np.array(test_i), np.array(test_r))
#train_iter = data.DataLoader(train_set, shuffle=True, last_batch='rollover', batch_size=batch_size)
#test_iter = data.DataLoader(test_set, batch_size=batch_size)