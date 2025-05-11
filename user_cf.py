import numpy as np
from load_data import ratings_matrix


# TODO: Add 'Top Rated Movies' fallback feature for new users
# Idea: For users with no history, recommend globally highest-rated movies.
# Strategy:
#   - Compute average rating per movie (from ratings_matrix)
#   - Sort descending and pick top-N
#   - Optionally add threshold: only consider movies with > X ratings





def cosine_similarity(user1, user2):
    """Compute cosine similarity between two user rating vectors."""
    #Step1 : Find indices where both users have rated
    common_indices = ~np.isnan(user1) & ~np.isnan(user2)
    
    # np.isnan() filters out any missing values (unrated movies) or values that have NaN as a value 
    
    
    if not np.any(common_indices):
        #No common ratings, similarity = 0
        return 0
    
    #Step 2 : extract the rating only for common movies
    user1_common = user1[common_indices]
    user2_common = user2[common_indices]

    #Step 3: Calculate cosine similarity
    numerator = np.dot(user1_common, user2_common)
    denominator = np.linalg.norm(user1_common) * np.linalg.norm(user2_common)

    if denominator == 0:
        return 0

    similarity = numerator / denominator
    return similarity

# user1 = np.array([5,5,5])
# user2 = np.array([1,1,1])
#pattern highly affect the value 
# i.e., even if the total rating are same but the pattern is different, it can highly alter the value of the rating finally obtained
#cosine similarity is about the direction rather than the magnitude. Basically the cosine of the angle formed when we represent the entities as a vector

# similarity_score = cosine_similarity(user1,user2)
# print(f"Cosine Similarity between User1 and user2: {similarity_score:.4f}")


num_users = ratings_matrix.shape[0]
user_similarity_matrix = np.zeros((num_users, num_users))

for i in range(num_users):
    for j in range(num_users):
        if i!=j:
            user_similarity_matrix[i][j] = cosine_similarity(
                ratings_matrix.iloc[i].values,
                ratings_matrix.iloc[j].values
            )
        else:
            user_similarity_matrix[i][j] = 1.0       #similarity with self

def get_top_k_similar_users(user_similarity_matrix, user_index, k=5):
    #extract similarity scores for the target user
    similarities = user_similarity_matrix[user_index]
    
    #Enumerate and sort other users by similarity (excluding self)
    similar_users = sorted(
        [(other_user, score) for other_user, score in enumerate(similarities) if other_user!=user_index],
        key = lambda x: x[1],
        reverse= True
    )
    return similar_users[:k]    #return top-k

# #test for user0
# user_index = 0
# top_k = get_top_k_similar_users(user_similarity_matrix, user_index, k=2)
# print(f"Top 2 similar users to user {user_index}: {top_k}")


def predict_rating_user_based(target_user,target_movie,rating_matrix, user_similarity_matrix,k=3):
    #Step1: Get similarityscores for all users wrt target_user
    similarities = user_similarity_matrix[target_user]
    
    #Step2: Enumerate them with user indices and sort by similarity
    similar_users = [(other_user, sim) for other_user, sim in enumerate(similarities) if other_user!=target_user]
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    #Step3: Take top-k similar users
    top_k_users = similar_users[:k]
    
    #Step4: Compute weighted avaerage of their ratings for the target movie
    numerator = 0.0
    denominator = 0.0
    
    for user, similarity in top_k_users:
        #rating = rating_matrix[user][target_movie]
        #rating = rating_matrix[user].get(target_movie, np.nan)
        rating = rating_matrix.iloc[user, target_movie]

        
        print(f"User {user} similarity: {similarity:.4f}, rating for movie {target_movie}: {rating}")
        if rating > 0:  #Consider only if the user has rated that movie
            numerator += similarity * rating
            denominator += similarity
            
    if denominator == 0:
        return 0    #No similar users have rated this movie
    else:
        return numerator
    
    # for user, similarity in top_k_users:
    #     rating = rating_matrix[user][target_movie]
    #     print(f"User {user} similarity: {similarity:.4f}, rating for movie {target_movie}: {rating}")

        
            
#Step 6: Test prediction
#Map user_id and item_id to matrix indices 
user_id_to_index = {user_id: idx for idx, user_id in enumerate(ratings_matrix.index)}
item_id_to_index = {item_id: idx for idx, item_id in enumerate(ratings_matrix.columns)}

#Example test
target_user_id = 1
target_movie_id = 50


user_idx = user_id_to_index[target_user_id]
movie_idx = item_id_to_index[target_movie_id]


predicted = predict_rating_user_based(user_idx, movie_idx, ratings_matrix.fillna(0), user_similarity_matrix, k=3)
print(f"Predicted rating by user {target_user_id} for movie {target_movie_id}: {predicted:.2f}")