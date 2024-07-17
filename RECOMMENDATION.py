import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv('path_to_ratings.csv')  
movies = pd.read_csv('path_to_movies.csv')  

data = pd.merge(ratings, movies, on='movieId')

user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

sparse_user_item_matrix = csr_matrix(user_item_matrix.values)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_user_item_matrix = train_data.pivot_table(index='userId', columns='title', values='rating')
train_user_item_matrix.fillna(0, inplace=True)
sparse_train_user_item_matrix = csr_matrix(train_user_item_matrix.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(sparse_train_user_item_matrix)

def get_movie_recommendations(user_id, n_recommendations):
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=n_recommendations+1)
    recommended_movies = [user_item_matrix.columns[indices.flatten()[i]] for i in range(1, n_recommendations+1)]
    return recommended_movies

user_id = 1
n_recommendations = 5
recommended_movies = get_movie_recommendations(user_id, n_recommendations)
print(f"Recommended movies for user {user_id}: {recommended_movies}")

def get_rmse(model, data):
    users, items, preds, actuals = [], [], [], []
    for user_id, movie_id, rating in zip(data['userId'], data['movieId'], data['rating']):
        user_index = user_id - 1
        movie_index = user_item_matrix.columns.get_loc(movies[movies['movieId'] == movie_id]['title'].values[0])
        user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = model.kneighbors(user_ratings, n_neighbors=2)
        pred_rating = user_item_matrix.iloc[indices.flatten()[1], movie_index]
        users.append(user_id)
        items.append(movie_id)
        preds.append(pred_rating)
        actuals.append(rating)
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    return rmse

rmse = get_rmse(model_knn, test_data)
print(f"Root Mean Squared Error: {rmse}")
