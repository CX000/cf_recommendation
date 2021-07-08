import pickle


# if __name__ == "__main__":

model_name = 'cf_model.pkl'    
input_json = {'user_id': 5, 'item_id': 10}

# load pre-trained pkl model
loaded_model = pickle.load(open(model_name, 'rb'))
rate, recommendations = loaded_model.predict(input_json)

print(f'Ratings of user_id {input_json["user_id"]} on movie_id {input_json["item_id"]}: {rate}')
print(f'Top 5 recommendations for user {input_json["user_id"]} is: {recommendations}')