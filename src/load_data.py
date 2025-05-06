import pandas as pd
# import numpy as np

#path to dataset 
ratings_path = "../ml-100k/u.data"
movies_path = "../ml-100k/u.item"

#load the dataset
#df = pd.read_csv(dataset_path)
#df = pd.read_csv("../data/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
#df = pd.read_csv(dataset_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(ratings_path, sep='\t', names=column_names)

#print(df.head())                      prints the top 5 values

#create user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating')
ratings_matrix = user_item_matrix    
#Replace all NaN values in user_item_matrix with 0
user_item_matrix_filled = user_item_matrix.fillna(0)

def load_movie_titles():
    #Load movie titles
    # u.item has pipe-separated values: movie_id | title | release_date | etc.
    movie_columns = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
    movies_df = pd.read_csv(movies_path, sep='|', names=movie_columns, encoding='latin-1')
    #stores the names of the movies to show the suggestions as movie names instead of movie ids.

    #Create a mapping from movie ID to tiltle
    return dict(zip(movies_df['movie_id'], movies_df['title']))

#print(movie_id_to_title.head())

# At the end of load_data.py
__all__ = ['ratings_matrix', 'user_item_matrix', 'movie_id_to_title']
