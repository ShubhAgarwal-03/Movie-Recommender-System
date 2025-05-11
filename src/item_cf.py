import numpy as np
import pandas as pd
#from load_data import ratings_matrix, movie_id_to_title        #already preprocessed user-item matrix(with NaNs)
from load_data import ratings_matrix, load_movie_titles
movie_id_to_title = load_movie_titles()


item_id_to_index = {item_id: idx for idx, item_id in enumerate(ratings_matrix.columns)}

index_to_movie_id = {idx: item_id for item_id, idx in item_id_to_index.items()}

ratings_matrix = pd.DataFrame(ratings_matrix)


# Compare ratings from users who have rated both items
def cosine_similarity(item1, item2):
    """Compute cosine similarity between 2 item rating vectors"""""
    common_indices = ~np.isnan(item1) & ~np.isnan(item2)
    if not np.any(common_indices):
        return 0
    
    item1_common = item1[common_indices]
    item2_common = item2[common_indices]
    
    #Calculation of cosinse similarity
    numerator = np.dot(item1_common, item2_common)
    denominator = np.linalg.norm(item1_common)* np.linalg.norm(item2_common)
    
    return numerator/denominator if denominator !=0 else 0

#Transpose the matrix to get item-user matrix
item_user_matrix = ratings_matrix.T     #shape: (num_items, num_users)

# Creation of a square matrix from the same 
num_items = item_user_matrix.shape[0]
item_similarity_matrix = np.zeros((num_items, num_items))

for i in range(num_items):
    for j in range(num_items):
        if i!=j:
            item_similarity_matrix[i][j] = cosine_similarity(
                item_user_matrix.iloc[i].values,
                item_user_matrix.iloc[j].values
            )
        else:
            item_similarity_matrix[i][j] = 1.0
            #set similarity to 1.0 for an item compared to itself

#If 2 items are rated similarly by the same users, they are considered 
# similar even if rated by few users


def predict_rating_item_based(user_index, target_item_index, ratings_matrix, item_similarity_matrix,index_to_movie_id, k=3):
    
    #print("Type of ratings_matrix:", type(ratings_matrix))

    
    #Step1: Get all the item ratings by the user
    #Step 1: Convert ratings_matrix to DataFrame if it's a NumPy array
    if isinstance(ratings_matrix, np.ndarray):
        ratings_matrix = pd.DataFrame(ratings_matrix)

    #ratings_matrix = ratings_matrix.fillna(0)
    # if isinstance(ratings_matrix, pd.DataFrame):
    #     ratings_matrix = ratings_matrix.fillna(0)
    # else:
    #     ratings_matrix = np.nan_to_num(ratings_matrix)
    ratings_matrix = ratings_matrix.fillna(0)
    user_ratings = ratings_matrix.iloc[user_index]
    
    #Step2: Find indices of items the user has rated
    #rated_items_indices = np.where(user_ratings > 0)[0]
    
    #rated_items_indices = user_ratings[user_ratings > 0][0]
    rated_items_indices = np.where(user_ratings > 0)[0]
       
    
    #Step3: Get similarity scores between the target item and the item the user has rated
    similarities_and_ratings = []
    
    for item_idx in rated_items_indices:
        #item_id = index_to_movie_id[item_idx]
        #item_idx = ratings_matrix.columns.get_loc(item_id)
        sim = item_similarity_matrix[target_item_index][item_idx]
        rating = user_ratings.iloc[item_idx]
        
        
        similarities_and_ratings.append((sim,rating))
        # Making a tuple of similarity and ratings, this will then be used to compare items(if required)
        
    #Step4: Sort by similarity and take top-k
    top_k = sorted(similarities_and_ratings, key=lambda x: x[0], reverse=True)[:k]
    
    #Step5: Compute weighted average
    numerator = sum(sim * rating for sim, rating in top_k)
    denominator = sum(abs(sim) for sim, _ in top_k)
    
    if denominator == 0:
        return 0
    
    return numerator/denominator
         
         
index_to_movie_id = {idx: item_id for item_id, idx in item_id_to_index.items()}


def recommend_items_item_based(user_id, ratings_matrix, item_similarity_matrix, user_id_to_index, item_id_to_index, k=3, n_recommendations = 5):  
    # user_id_to_index = {uid: idx for idx, uid in enumerate(ratings_matrix.index)}
    # item_id_to_index = {iid: idx for idx, iid in enumerate(ratings_matrix.columns)}
    
    user_idx = user_id_to_index[user_id]
    user_ratings = ratings_matrix.iloc[user_idx].values
    
    unrated_items = np.where(np.isnan(user_ratings))[0]
    
    predictions = [] 
    for item_idx in unrated_items:
        predicted_rating = predict_rating_item_based(user_idx, item_idx, ratings_matrix.fillna(0), item_similarity_matrix, index_to_movie_id, k)
        predictions.append((item_idx, predicted_rating))
        #This line is storing a tuple for each movie the user hasn’t rated yet.
        
    #Sort by predicted rating 
    top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
    #The [:n_recommendations] selects only the top-N items from the list using slicing
    
    #Convert back to actual item IDs
    index_to_item_id = {idx: item_id for item_id, idx in item_id_to_index.items() }
    recommend_items = [(index_to_item_id[item_idx], rating) for item_idx, rating in top_items]
    # Matrix uses item_idx (numerical indices to comparison) but we want to return movie id
    # Pehele we convert the item_id to index and then do the comparions. 
    # Now since all the comparisons are done, we need to convert them back to id to return to the user. This is done in the index_to_item_id
    
    
    # actual_movie_id = index_to_item_id[item_idx]
    # user_ratings[actual_movie_id]


    
    return recommend_items        



user_id_to_index = {uid: idx for idx, uid in enumerate(ratings_matrix.index)}
item_id_to_index = {iid: idx for idx, iid in enumerate(ratings_matrix.columns)}


#recommendations = recommend_items_item_based(user_id=1, ratings_matrix=ratings_matrix, item_similarity_matrix=item_similarity_matrix)
recommendations = recommend_items_item_based(
    user_id=1,
    ratings_matrix=ratings_matrix,
    item_similarity_matrix=item_similarity_matrix,
    user_id_to_index=user_id_to_index,
    item_id_to_index=item_id_to_index
)

print("Top recommendations for User 1 (item-based):")
for movie_id, predicted_rating in recommendations:
    title = movie_id_to_title.get(movie_id, "Unknown Title")
    print(f"{title}(Movie ID {movie_id}) → Predicted Rating: {predicted_rating:.2f}")





         
# #Step 6: Test prediction
# #Map user_id and item_id to matrix indices 
# user_id_to_index = {user_id: idx for idx, user_id in enumerate(ratings_matrix.index)}
# item_id_to_index = {item_id: idx for idx, item_id in enumerate(ratings_matrix.columns)}


# user_id = 1
# movie_id = 50

# user_idx = user_id_to_index[user_id]
# item_idx = item_id_to_index[movie_id]

# predicted_rating = predict_rating_item_based(user_idx, item_idx, ratings_matrix.fillna(0).values, item_similarity_matrix)
# print(f"Predicted rating by user {user_id} for movie {movie_id} (item-based): {predicted_rating:.2f}")
