import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_rows', 20)


class ContentRecSys:
    def __init__(self):
        missing_values = ['na', '--', '?', '-', 'None', 'none', 'non']
        self.movies_df = pd.read_csv('data/ml-1m/movies_new.dat', sep=':', names=['movie_id', 'title', 'genres'], na_values=missing_values)
        self.ratings_df = pd.read_csv('data/ml-1m/ratings_new.dat', sep=':', names=['user_id', 'movie_id', 'rating', 'timestamp'], na_values=missing_values)

    def data_prep(self):
        # Using regular expressions to find a year stored between parentheses
        # We specify the parentheses so we don't conflict with movies that have years in their titles.
        self.movies_df['year'] = self.movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
        # Removing the parentheses.
        # Note that expand=False simply means do not add this adjustment as an additional column to the data frame.
        self.movies_df['year'] = self.movies_df.year.str.extract('(\d\d\d\d)', expand=False)
        # Removing the years from the 'title' column.
        self.movies_df['title'] = self.movies_df.title.str.replace('(\(\d\d\d\d\))', '')
        # Applying the strip function to get rid of any ending white space characters that may have appeared, using lambda function.
        self.movies_df['title'] = self.movies_df['title'].apply(lambda x: x.strip())
        # Every genre is separated by a | so we simply have to call the split function on |.
        self.movies_df['genres'] = self.movies_df.genres.str.split('|')
        # Checking for missing values.
        # print(movies_df.isna().sum())
        # Filling year NaN values with zeros.
        self.movies_df.year.fillna(0, inplace=True)
        # Converting columns year from obj to int16 and movieId from int64 to int32 to save memory.
        self.movies_df.year = self.movies_df.year.astype('int16')
        self.movies_df.movie_id = self.movies_df.movie_id.astype('int32')
        # print(movies_df.info(), '\n', movies_df.head())
        # First let's make a copy of the movies_df.
        movies_with_genres = self.movies_df.copy(deep=True)
        # Let's iterate through movies_df, then append the movie genres as columns of 1s or 0s.
        x = []
        for index, row in self.movies_df.iterrows():
            x.append(index)
            for genre in row['genres']:
                movies_with_genres.at[index, genre] = 1
        # Confirm that every row has been iterated and acted upon.
        # print(len(x) == len(movies_df))
        # Filling in the NaN values with 0 to show that a movie doesn't have that column's genre.
        movies_with_genres = movies_with_genres.fillna(0)
        # print(movies_with_genres.head(3))
        # Dropping the timestamp column to save memory
        self.ratings_df.drop('timestamp', axis=1, inplace=True)
        # print(ratings_df.head(3))
        return movies_with_genres

    def recommend_movies(self, user_movie_data, movies_with_genres, nb_rec=20):
        # Step1: Creating User's profile
        # so on a scale of 0 to 5, with 0 min and 5 max, see user's movie ratings below.
        # create a DataFrame if user_movie_data is not already one
        user_movie_ratings = pd.DataFrame(user_movie_data)
        # Extracting movie Ids from movies_df and updating user_movie_ratings with movie Ids.
        # user_movie_id = movies_df[movies_df['title'].isin(user_movie_ratings['title'])]  # in case using title as input in user ratings
        user_movie_id = self.movies_df[self.movies_df['movie_id'].isin(user_movie_ratings['movie_id'])] # in case using movie_id as input in user ratings
        # Merging user movie_id and ratings into the user_movie_ratings data frame.
        # This action implicitly merges both data frames by the title column.
        user_movie_ratings = pd.merge(user_movie_id, user_movie_ratings)
        # Dropping information we don't need such as year and genres.
        user_movie_ratings = user_movie_ratings.drop(['genres', 'year'], 1)
        print(user_movie_ratings)

        # Step 2: Learning User’s Profile
        # filter the selection by outputing movies that exist in both user_movie_ratings and movies_with_genres.
        user_genres_df = movies_with_genres[movies_with_genres.movie_id.isin(user_movie_ratings.movie_id)]
        # First, let's reset index to default and drop the existing index.
        user_genres_df.reset_index(drop=True, inplace=True)
        # Next, let's drop redundant columns
        user_genres_df.drop(['movie_id', 'title', 'genres', 'year'], axis=1, inplace=True)
        # print(user_genres_df)

        # Step 3: Building User’s Profile
        # let's confirm the shapes of our data frames to guide us as we do matrix multiplication.
        # print('Shape of user_movie_ratings is:', user_movie_ratings.shape)
        # print('Shape of user_genres_df is:', user_genres_df.shape)
        # Let's find the dot product of transpose of user_genres_df by Lawrence rating column.
        user_profile = user_genres_df.T.dot(user_movie_ratings.rating)  # Let's see the result
        # print(user_profile)

        # Step 4: Deploying The Content-Based Recommender System.
        # let's set the index to the movieId.
        movies_with_genres = movies_with_genres.set_index(movies_with_genres.movie_id)
        # Deleting four unnecessary columns.
        movies_with_genres.drop(['movie_id', 'title', 'genres', 'year'], axis=1, inplace=True)
        # print(movies_with_genres.head())
        # Multiply the genres by the weights and then take the weighted average.
        recommendation_table_df = (movies_with_genres.dot(user_profile)) / user_profile.sum()
        # Let's sort values from great to small
        recommendation_table_df.sort_values(ascending=False, inplace=True)
        # print(recommendation_table_df.head())
        # first we make a copy of the original movies_df
        copy = self.movies_df.copy(deep=True)
        # Then we set its index to movieId
        copy = copy.set_index('movie_id', drop=True)
        # Next we enlist the top n recommended movie_ids we defined above
        top_n_index = recommendation_table_df.index[:nb_rec].tolist()
        # finally we slice these indices from the copied movies df and save in a variable
        recommended_movies = copy.loc[top_n_index, :]
        # Now we can display the top 20 movies in descending order of preference
        # print(recommended_movies)
        # return recommended_movies # this will return the list of movies (titles) with year and genres
        return pd.DataFrame(recommended_movies.index) # this will return the indexes (representing movie_ids) of the dataframe


if __name__ == "__main__":
    # if we want to create manually a user's profile:
    user_movie_data = [
        {'movie_id': 3459, 'rating': 4.9},
        {'movie_id': 3341, 'rating': 4.9},
        {'movie_id': 1802, 'rating': 4},
        {'movie_id': 2546, 'rating': 3},
    ]
    recommender = ContentRecSys()
    movies_with_genres = recommender.data_prep()
    recommended_movies = recommender.recommend_movies(user_movie_data, movies_with_genres, 20)
    print(recommended_movies)
