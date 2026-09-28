# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random

# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or 
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.
    
    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix), 
                    1: claculate similarity scores between items \
                        (columns of the matrix)
    
    Returns:
        similarity_dict (dictionary): dictionary where the keys are users 
        (or items) and the values are lists of tuples containing the most 
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to authomatically compute cosine similarity using
    an apposite function from any package, the computation should follow the 
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for 
    faster computation.
    """
    similarity_dict= {}
    # TO DO: Handle the absence of ratings (missing values in the matrix)
    matrix = matrix.fillna(0)  # for the cosine similarity to calculate we cannot use NaN values so we put the missing values to 0



    # TO DO: If axis is 1, what do you need to do to calculate the similarity 
    # between items (columns)
    if axis==1:
        matrix = matrix.T    # so for axis=1 we need to use items instead of user so we transpose the matrix to change the user and item shapes

    # TO DO: loop through each couple of entities to calculate their cosine 
    # similarity and store these results

    matrix_new = matrix.to_numpy() # converting pandas into numpy for dot product calculation

    for i in range(matrix_new.shape[0]):  # going through all users or items
        list=[]  # list to keep similarity scores
        for j in range(matrix_new.shape[0]):
            if i == j:
                continue
            user_vector_i = matrix_new[i]           # for example user 1
            user_vector_j = matrix_new[j]           # then all the other users except 1
            numerator = np.dot(user_vector_i, user_vector_j)  # cosine similarity formula
            denominator = (np.sqrt(np.sum(user_vector_i ** 2)))*(np.sqrt(np.sum(user_vector_j ** 2)))   # cosine similarity formula
            if denominator == 0:
                similarity = 0
            else:
                similarity = numerator / denominator    # cosine similarity formula
            list.append((j,similarity))       # updating list
        list.sort(key=lambda x: x[1], reverse=True)   # sorting list such that x[1] that is the similarity score is taken into account when sorting
        similarity_dict[matrix.index[i]] = list[:k]   # the list is sorted from greater to smaller so taking the first k items

    # TO DO: sort the similarity scores for each entity and add the top k most 
    # similar entities to the similarity_dict


    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    similar_users = user_similarity[user_id][:k]

    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0

    for s_user_id, similarity in similar_users:                    # for every similar user and their similarity
        rating = user_item_matrix.loc[s_user_id,movie_id]           # getting the movie rating of the similar user

        if pd.notna(rating):     # checking for undefined values, because we only want the user if they rated the item
            numerator += similarity * rating    # formula
            denominator += similarity


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target item
    similar_items = item_similarity[movie_id][:k]


    # TO DO: implement item-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0

    for s_item_id, similarity in similar_items:                    # for every similar item and their similarity
        rating = user_item_matrix.loc[user_id,s_item_id]           # getting the user rating of the similar item

        if pd.notna(rating):   # checking for undefined values, because we only want the item if they rated bt the user
            numerator += similarity * rating       # formula
            denominator += similarity


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating

# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    for step in range(n_steps):
        # TODO: Implement the algorithm to update user_matrix and item_matrix
        for user in range(utility_matrix.shape[0]):
            for item in range(utility_matrix.shape[1]):     # going through every user and item

                if utility_matrix[user, item] > 0:  # only updating for ratings that are defined

                    err = (utility_matrix[user, item]) - (np.dot(user_matrix[user], item_matrix[item]))  # error formula

                    i_vec = item_matrix[item].copy()  # vector copy as it is updated later and we have to use the current vector

                    item_matrix[item] += learning_rate * (err * user_matrix[user] - regularization * item_matrix[item])  # update formula

                    user_matrix[user] += learning_rate * (err * i_vec - regularization * user_matrix[user])  # update formula


    return user_matrix, item_matrix

if __name__ == "__main__":
    path = "u.data"       
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13  
    movie_id = 100  

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
