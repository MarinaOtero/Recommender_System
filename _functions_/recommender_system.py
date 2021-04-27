#!/usr/bin/env python
# coding: utf-8

# # Ratings

# In[]:


import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, load_npz
from sklearn.neighbors import NearestNeighbors


# In[]:


ratings = pd.read_csv('../clean_data/ratings_gender.csv') # Los usuarios con menos de 15 reseñas han sido eliminados

#Lower title and Director Name
ratings['primaryName'] = ratings['primaryName'].str.lower()
ratings['primaryTitle'] = ratings['primaryTitle'].str.lower()
# From string to tuple
ratings['tconst_gender'] = ratings['tconst_gender'].apply(lambda x: eval(x))

N = ratings['userID'].nunique()
M = ratings['tconst_gender'].nunique()
    

user_mapper = dict(zip(np.unique(ratings["userID"]), list(range(N))))
movie_mapper = dict(zip(np.unique(ratings["tconst_gender"]), list(range(M))))
    
user_inv_mapper = dict(zip(list(range(N)), np.unique(ratings["userID"])))
movie_inv_mapper = dict(zip(list(range(M)), np.unique(ratings["tconst_gender"])))
    
user_index = [user_mapper[i] for i in ratings['userID']]
movie_index = [movie_mapper[i] for i in ratings['tconst_gender']]

#MODEL
kNN = pickle.load(open('../_model_/KNN.sav', 'rb'))

#Sparse matrix 
X = load_npz('../clean_data/sparse_matrix.npz')


#In[8]:


# ## Finding similar movies using k-Nearest Neighbours
# 
# This approach looks for the $k$ nearest neighbours of a given movie by identifying $k$ points in the dataset that are closest to movie $m$. kNN makes use of distance metrics such as:
# 
# 1. Cosine similarity
# 2. Euclidean distance
# 3. Manhattan distance
# 4. Pearson correlation 
# 
# Although difficult to visualize, we are working in a M-dimensional space where M represents the number of movies in our X matrix. 

# In[5]:




def find_similar_movies(movie_title, k=500, show_distance=False):
    """
    Finds k-nearest neighbours for a given movie id.
    
    Args:
        movie_id: id of the movie of interest
        X: user-item utility matrix
        k: number of similar movies to retrieve (por ahora solo va a devolver 5)
        metric: distance metric for kNN calculations
    
    Returns:
        list of k similar movie ID's
    """
    movie_id = ratings.tconst_gender[ratings.primaryTitle == movie_title].unique()[0]

    neighbour_ids = []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    
    
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    
    i = 0
    j = 0
    while i < 5:
        n = neighbour.item(j)
        j = j + 1
        if movie_inv_mapper[n][1] == 'F':
            neighbour_ids.append(movie_inv_mapper[n])
            i = i + 1
        if j==k:
            break
            
    neighbour_ids.pop(0)
    return neighbour_ids


