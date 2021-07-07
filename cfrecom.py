import pickle
import os
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds


class cf_recom:
    
    def __init__(self, movie_path, rating_path):
        self.movie_path = movie_path
        self.rating_path = rating_path
        self.movies_df = None
        
    def prep_data(self):
        """
        Function used to prepare data
        """
        self.movies_df = pd.read_csv(self.movie_path, 
                                usecols=['movieId', 'title'],
                                dtype={'movieId': 'int32', 'title': 'str'})
        ratings_df = pd.read_csv(self.rating_path,
                                usecols=['userId', 'movieId', 'rating'],
                                dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.ratings_df = ratings_df[:20000]  # used only first 2M records for testing
        rating_pivot = self.ratings_df.pivot(index='userId', 
                                            columns='movieId', 
                                            values='rating').fillna(0)
        ratings = rating_pivot.values
        user_ratings_mean = np.mean(ratings, axis = 1)
        rating_demeaned = ratings - user_ratings_mean.reshape(-1, 1)
        return rating_pivot, user_ratings_mean, rating_demeaned
    
    def train(self):
        """
        Function used to calculate User and Item matrix using SVD, and predict rating matrix
        """
        rating_pivot, user_ratings_mean, rating_demeaned = self.prep_data()
        U, sigma, Vt = svds(rating_demeaned, k = 50)
        sigma = np.diag(sigma)
        self.predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        self.preds_df = pd.DataFrame(self.predicted_ratings, columns=rating_pivot.columns)
        # return predicted_ratings, preds_df
    
    def predict(self, input_json, num_recommendations=5):
        """
        Function used to predict ratings of a user on an item, and recommendate first 5 highest rating movies
        """
        user_id = input_json['user_id']
        item_id = input_json["item_id"]
        rate = 0
        if item_id != -1:
            rate = round(self.preds_df.iloc[user_id-1].iloc[item_id -1],2)
        sorted_user_predictions = self.preds_df.iloc[user_id-1].sort_values(ascending=False) # UserID starts at 1
        user_row_number = user_id - 1
        user_data = self.ratings_df[self.ratings_df.userId == (user_id)]
        user_full = (user_data.merge(self.movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                        sort_values(['rating'], ascending=False))
        recommendations = (self.movies_df[~self.movies_df['movieId'].isin(user_full['movieId'])]).merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'movieId',
                right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
        return rate, recommendations
    
    # def save(self, file_name):
    #     """Save thing to a file."""
    #     f = open(file_name,"w")
    #     pickle.dump(self,f)
    #     f.close()
    
    
if __name__ == "__main__":
    
    from cfrecom import cf_recom
    movies_filepath = os.path.join(os.getcwd(),'movie_recom/movie.csv')
    ratings_filepath = os.path.join(os.getcwd(),'movie_recom/rating.csv')
    cf = cf_recom(movies_filepath, ratings_filepath)
    cf.train()
    test_json = {'user_id': 5, 'item_id': 10}
    rate, recommendations = cf.predict(test_json)
    
    model_name = 'cf_model1.pkl'
    pickle.dump(cf, open(model_name, 'wb')) # save model as pkl file
    