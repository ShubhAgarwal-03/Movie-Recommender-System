import numpy as np
import pandas as pd
from load_data import ratings_matrix, load_movie_titles
from user_cf import predict_rating_user_based, user_similarity_matrix
from item_cf import predict_rating_item_based, item_similarity_matrix



#Map user_id and item_id to matrix indices 
user_id_to_index = {user_id: idx for idx, user_id in enumerate(ratings_matrix.index)}
item_id_to_index = {item_id: idx for idx, item_id in enumerate(ratings_matrix.columns)}


index_to_movie_id = {idx: item_id for item_id, idx in item_id_to_index.items()}

ratings_matrix = pd.DataFrame(ratings_matrix)


movie_id_to_title = load_movie_titles()

def hybrid_predict(user_idx, item_idx, ratings_matrix, user_similarity_matrix, item_similarity_matrix,index_to_movie_id, alpha=0.5):
    
    #user_id_to_index, item_id_to_index
    """ 
    Combine user-based and item-based predictions using a weighted average 
    """
    # user_based_rating = user_based_func(user_idx, item_idx, ratings_matrix, user_similarity_matrix)
    # item_based_rating = item_based_func(user_idx, item_idx, ratings_matrix, item_similarity_matrix)
    
    user_based_rating = predict_rating_user_based(user_idx, item_idx, ratings_matrix, user_similarity_matrix)
    item_based_rating = predict_rating_item_based(user_idx, item_idx, ratings_matrix, item_similarity_matrix, index_to_movie_id, k=3)
    
    
    #user_idx = user_id_to_index[user_id]
    #item_idx = item_id_to_index[item_id]
    
    #user_pred = user_based_func(user_idx, item_idx, ratings_matrix, user_similarity_matrix)
    #item_pred = item_based_func(user_idx, item_idx, ratings_matrix, item_similarity_matrix)
    if user_based_rating == 0 and item_based_rating == 0:
        return 0.0
    elif user_based_rating == 0.0:
        return item_based_rating
    elif item_based_rating == 0.0:
        return user_based_rating
    else:
        return alpha*user_based_rating + (1-alpha)*item_based_rating
    
    
user_idx = 1
item_idx = 50
alpha = 0.5
    
hybrid_rating = hybrid_predict (
    user_idx=user_idx,
    item_idx=item_idx,
    ratings_matrix = ratings_matrix.fillna(0),
    #user_id_to_index = user_id_to_index,
    #item_id_to_index = item_id_to_index,
    #user_based_func = predict_rating_user_based,
    #item_based_func = predict_rating_item_based,
    user_similarity_matrix=user_similarity_matrix,
    item_similarity_matrix=item_similarity_matrix,
    index_to_movie_id = index_to_movie_id,
    alpha = alpha
    )
print(f"Hybrid predicted rating for user {user_idx} on movie {item_idx}: {hybrid_rating:.2f}")



#def recommend_items_hybrid(user_id, ratings_matrix, user_id_to_index, user_similarity_matrix, item_similarity_matrix, movie_id_to_title, alpha=0.5, n_recommendations=5):
 
def recommend_items_hybrid(user_id, ratings_matrix, user_id_to_index, item_id_to_index,
                           user_similarity_matrix, item_similarity_matrix, movie_id_to_title,
                           index_to_movie_id,   
                           alpha=0.5, n_recommendations=5):

    user_idx = user_id_to_index[user_id]
    
    predictions = []
    for item_id in ratings_matrix.columns:
        if item_id not in item_id_to_index:
            continue    #skip unknown items
        
        item_idx = item_id_to_index[item_id]
        try:
            if pd.isna(ratings_matrix.loc[user_id, item_id]):
                pred = hybrid_predict(user_idx, item_idx, ratings_matrix, 
                                      #user_based_func = predict_rating_user_based,
                                      #item_based_func = predict_rating_item_based,
                                      user_similarity_matrix = user_similarity_matrix,
                                      item_similarity_matrix = item_similarity_matrix,
                                      index_to_movie_id = index_to_movie_id,
                                      #user_id_to_index = user_id_to_index,
                                      #item_id_to_index = item_id_to_index,
                                      alpha = alpha
                                      )
                predictions.append((item_id, pred))
        except KeyError:
            continue    #skip movies with indexing issues
            
    #Sort predictions by ratings
    top_items = sorted(predictions, key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    #Return movie titles instead of ids
    recommend_items = [(movie_id_to_title.get(item_id, f"Movie ID {item_id}"), rating) for item_id, rating in top_items]
    return recommend_items

recommendations = recommend_items_hybrid(
    user_id = 1,
    ratings_matrix = ratings_matrix,
    user_id_to_index = user_id_to_index,
    item_id_to_index = item_id_to_index,
    user_similarity_matrix = user_similarity_matrix,
    item_similarity_matrix = item_similarity_matrix,
    movie_id_to_title = movie_id_to_title,
    index_to_movie_id = index_to_movie_id,
    alpha = 0.5,
    n_recommendations = 5
)

print("\nTop Hybrid Recommendations for User 1:")
for title, rating in recommendations:
    print(f"{title} → Predicted Rating: {rating:.2f}")
