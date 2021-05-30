# A simple implementation of matrix factorization for collaborative filtering expressed as a Keras Sequential model

# Keras uses TensorFlow tensor library as the backend system to do the heavy compiling

import numpy as np
# from keras.layers import Embedding, Reshape, Merge
# from keras.models import Sequential
from keras.models import Model, Sequential
from keras.layers import Embedding, Flatten, Input, Dropout, Dense, BatchNormalization, Reshape, Dot

class CFModel(Sequential):

    # The constructor for the class
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        # P is the embedding layer that creates an User by latent factors matrix.
        # If the intput is a user_id, P returns the latent factor vector for that user.
        P = Sequential()
        P.add(Embedding(input_dim=n_users, output_dim=k_factors, input_length=1))  # input_length is the sequence length
        P.add(Reshape((k_factors,)))

        # Q is the embedding layer that creates a Movie by latent factors matrix.
        # If the input is a movie_id, Q returns the latent factor vector for that movie.
        Q = Sequential()
        Q.add(Embedding(input_dim=m_items, output_dim=k_factors, input_length=1))  # input_length is the sequence length
        Q.add(Reshape((k_factors,)))

        super(CFModel, self).__init__(**kwargs)

        # Add more

        # Take the dot product of user and movie latent factor vectors to return the corresponding rating
        self.add(Dot([P, Q], axes=1))

    # The rate function to predict user's rating of unrated items
    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]
		
		
class NCFModel():

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        '''
        # MLP Embeddings
        movie_vec_mlp = Sequential()
        movie_vec_mlp.add(Embedding(input_dim=m_items, output_dim=k_factors, input_length=1, input_shape=[1]))
        movie_vec_mlp.add(Reshape((k_factors,)))

        user_vec_mlp = Sequential()
        user_vec_mlp.add(Embedding(input_dim=n_users, output_dim=k_factors, input_length=1, input_shape=[1]))
        user_vec_mlp.add(Reshape((k_factors,)))

        # MF Embeddings
        movie_vec_mf = Sequential()
        movie_vec_mf.add(Embedding(input_dim=m_items, output_dim=k_factors, input_length=1, input_shape=[1]))
        movie_vec_mf.add(Reshape((k_factors,)))

        user_vec_mf = Sequential()
        user_vec_mf.add(Embedding(input_dim=n_users, output_dim=k_factors, input_length=1, input_shape=[1]))

        user_vec_mf.add(Reshape((k_factors,)))
        # MLP layers
        model_mlp = Sequential()
        model_mlp.add(Merge([movie_vec_mlp, user_vec_mlp], mode='concat', dot_axes=1))
        model_mlp.add(Dropout(0.2))
        model_mlp.add(Dense(100, activation='relu'))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Dropout(0.2))
        model_mlp.add(Dense(50, activation='relu'))
        model_mlp.add(BatchNormalization())
        model_mlp.add(Dropout(0.2))
        model_mlp.add(Dense(10, activation='relu'))

        # Prediction from both layers
        model_mf = Sequential()
        model_mf.add(Merge([movie_vec_mf, user_vec_mf], mode='dot', dot_axes=1))
        self.model_mlp_mf = Sequential()
        self.model_mlp_mf.add(Merge([model_mf, model_mlp], mode='concat', dot_axes=1))

        # Final prediction
        self.model_mlp_mf.add(Dense(1, activation='relu'))
        '''

        # Define inputs
        movie_input = Input(shape=[1], name='movie-input')
        user_input = Input(shape=[1], name='user-input')

        # MLP Embeddings
        movie_embedding_mlp = Embedding(m_items, k_factors, name='movie-embedding-mlp')(movie_input)
        movie_vec_mlp = Flatten(name='flatten-movie-mlp')(movie_embedding_mlp)

        user_embedding_mlp = Embedding(n_users, k_factors, name='user-embedding-mlp')(user_input)
        user_vec_mlp = Flatten(name='flatten-user-mlp')(user_embedding_mlp)

        # MF Embeddings
        movie_embedding_mf = Embedding(m_items, k_factors, name='movie-embedding-mf')(movie_input)
        movie_vec_mf = Flatten(name='flatten-movie-mf')(movie_embedding_mf)

        user_embedding_mf = Embedding(n_users, k_factors, name='user-embedding-mf')(user_input)
        user_vec_mf = Flatten(name='flatten-user-mf')(user_embedding_mf)

        # MLP layers
        concat = Concatenate([movie_vec_mlp, user_vec_mlp], name='concat')
        concat_dropout = Dropout(0.2)(concat)
        fc_1 = Dense(100, name='fc-1', activation='relu')(concat_dropout)
        fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
        fc_1_dropout = Dropout(0.2)(fc_1_bn)
        fc_2 = Dense(50, name='fc-2', activation='relu')(fc_1_dropout)
        fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
        fc_2_dropout = Dropout(0.2)(fc_2_bn)

        # Prediction from both layers
        pred_mlp = Dense(10, name='pred-mlp', activation='relu')(fc_2_dropout)
        pred_mf = Dot([movie_vec_mf, user_vec_mf], name='pred-mf')
        combine_mlp_mf = Concatenate([pred_mf, pred_mlp], name='combine-mlp-mf')

        # Final prediction
        result = Dense(1, name='result', activation='relu')(combine_mlp_mf)

        self.model_final = Model([user_input, movie_input], result)

    def rate(self, user_id, item_id):
        return self.model_final.predict([np.array([user_id]), np.array([item_id])])[0][0]
