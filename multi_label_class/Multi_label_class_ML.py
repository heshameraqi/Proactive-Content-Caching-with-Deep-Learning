# mlp for multi-label classification
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss
from numpy import asarray
import numpy as np
import keras
from math import floor
import tensorflow as tf
from keras import backend as K

# to configure a neural network model for multi-label classification, we need:
#     Number of nodes in the output layer matches the number of labels.
#     Sigmoid activation for each node in the output layer.
#     Binary cross-entropy loss function.


def get_dataset():
    # users_profiles_file = open("Users_profiles_test.txt", "r")
    users_profiles_file = open("Users_profiles.txt", "r")
    contents = users_profiles_file.readlines()
    X = list()
    for c in contents:
        gender = int(c.split('\t')[1])
        age = int(c.split('\t')[2])
        occupation = int(c.split('\t')[3])
        zip_code = int(c.split('\t')[4])
        user_profile = [gender, age, occupation, zip_code]
        X.append(user_profile)
    users_profiles_file.close()
    # movies_ratings_file = open("MovieLens_rating_test.txt", "r")
    movies_ratings_file = open("MovieLens_rating.txt", "r")
    # nb_users, nb_movies = 10, 5
    contents = movies_ratings_file.readlines()
    list_users, list_movies = list(), list()
    for c in contents:
        user_id = int(c.split('\t')[0])
        movie_id = int(c.split('\t')[1])
        list_users.append(user_id)
        list_movies.append(movie_id)
    movies_ratings_file.close()
    nb_users = len(list(dict.fromkeys(list_users)))
    nb_movies = len(list(dict.fromkeys(list_movies)))
    Y = np.zeros((nb_users, nb_movies))
    # Y = np.zeros((nb_users, 3952))
    for i in range(0, len(list_users)):
        user_id = list_users[i]
        movie_id = list_movies[i]
        Y[user_id - 1, movie_id - 1] = 1
    print("nb_users: ", nb_users, " / nb_movies: ", nb_movies)
    X = np.array(X)
    # print(X)
    # print(Y)
    return X, Y


# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    # n_inputs represents the nb of features of the input data and n_outputs the nb of labels in the output
    model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", "binary_accuracy"])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", "binary_accuracy", "Recall"])
    return model


#'''
# evaluate a model using repeated k-fold cross-validation
def evaluate_model_bis(X, Y):
    results = list()
    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        Y_train, Y_test = Y[train_ix], Y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, Y_train, verbose=0, epochs=100)
        # make a prediction on the test set
        Y_hat = model.predict(X_test)
        # round probabilities to class labels
        Y_hat = Y_hat.round()
        # calculate accuracy
        acc = accuracy_score(Y_test, Y_hat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
        print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
    return results
#'''


# evaluate model using classic approac
# def evaluate_model(X, Y):
def evaluate_model(X, Y, X_test, Y_test):
    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(X, Y, verbose=1, epochs=50, validation_split=0.1, batch_size=2, shuffle=True)
    y_hat = model.predict(X_test)
    print('Predicted (probabilities): %s' % y_hat)
    y_hat_classes = np.round(y_hat)
    print('Actual classes: ', Y_test)
    print('Predicted (classes): %s' % y_hat_classes)
    # acc = accuracy_score(Y_test, y_hat_classes)
    # print('accuracy (exact match): %.3f' % acc)
    nb_hitlabels = np.sum(Y_test == y_hat_classes)
    acc2 = nb_hitlabels/Y_test.size
    print('accuracy (per label) (equivalent to Hamming loss): %.3f' % acc2)
    nb_hitlabels = np.sum((Y_test == y_hat_classes) & (Y_test == 1))
    acc3 = nb_hitlabels / np.sum(y_hat_classes == 1)
    print('accuracy (ratio of how much true positive where guessed compared to the total list of predicted films): %.3f' % acc3)
    acc4 = nb_hitlabels / (np.sum(Y_test == 1) + np.sum(y_hat_classes == 1) - nb_hitlabels)
    print('accuracy (ratio of how much true positive where guessed compared to the acutual list of movies in addition to missmatched movie guesses): %.3f' % acc4)


if __name__ == "__main__":
    X, Y = get_dataset()
    l_X = floor(len(X)*0.9)  # to define what portion goes to training + test in the model
    l_Y = floor(len(Y)*0.9)
    # evaluate model using k-fold cross-validation
    # results = evaluate_model_bis(X, Y)
    # evaluate model using classic approach
    evaluate_model(X[0:l_X], Y[0:l_Y], X[l_X:], Y[l_Y:])
    # movies_list = np.where(np.all(Y[0:0] == 1, axis=0))[0] + 1
    # print(movies_list)

