#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds  #for sparse matrices
import json
import matplotlib.image as mpimg
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score



#SIMILAR USERS

# Cosine Similarity
def similar_users(user_index, interactions_matrix):
    similarity = []

    for user in range(0, interactions_matrix.shape[0]):  
        
        # finding cosine similarity between the user_id and each user
        sim = cosine_similarity([interactions_matrix.loc[user_index]], [interactions_matrix.loc[user]])
        similarity.append((user, sim))

    similarity.sort(key=lambda x: x[1], reverse=True)
    most_similar_users = [tup[0] for tup in similarity]   
    similarity_score = [tup[1] for tup in
                        similarity]  

    # Remove the original user and its similarity score 
    most_similar_users.remove(user_index)
    similarity_score.remove(similarity_score[0])

    return most_similar_users, similarity_score


# Collaborative Filtering (user-based)
def recommendations(user_index, num_of_products, interactions_matrix, merged_df):
    
    most_similar_users = similar_users(user_index, interactions_matrix)[0]

    # Finding product ids with which the user_id has interacted
    prod_ids = set(list(interactions_matrix.columns[np.where(interactions_matrix.loc[user_index] > 0)]))

    recommendations = []

    observed_interactions = prod_ids.copy()
    for similar_user in most_similar_users:
        if len(recommendations) < num_of_products:
            # Finding n products which have been rated by similar users but not by the user_id
            similar_user_prod_ids = set(
                list(interactions_matrix.columns[np.where(interactions_matrix.loc[similar_user] > 0)]))
            recommendations.extend(list(similar_user_prod_ids.difference(observed_interactions)))
            observed_interactions = observed_interactions.union(similar_user_prod_ids)
        else:
            break

    selected_products = recommendations[:num_of_products]

    return selected_products

# Getting the url, title and ratings of the products to demonstrate on the website
def show(product_id, merged_df):

    selected_product = merged_df[merged_df['parent_asin'] == product_id].iloc[0]
    first_image_info = selected_product['images'][0]
    first_image_title = selected_product['title']
    rating = selected_product['rating']

    if 'large' in first_image_info:
        url = first_image_info['large']
        return {
            'title': first_image_title,
            'url': url,
            'rating': rating
        }


# Recommendations for T-SVD predictions
def recommend_items(user_index, interactions_matrix, preds_matrix, final_ratings_matrix, num_recommendations,
                    merged_df):

    user_ratings = interactions_matrix[user_index, :].toarray().reshape(-1)
    user_predictions = preds_matrix[user_index, :].toarray().reshape(-1)

    items = final_ratings_matrix.columns

    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = items

    temp = temp.loc[temp.user_ratings == 0]

    temp = temp.sort_values('user_predictions',
                            ascending=False)  
    
    selected_products = temp.iloc[:num_recommendations]

    return selected_products['Recommended Products']


# SIMILAR PRODUCTS
def similar_items(item_id, interactions_matrix):
    similarity = []
    item_vector = interactions_matrix[item_id].values.reshape(1, -1)
    
    item_vectors = interactions_matrix.values.T
    
    sim_scores = cosine_similarity(item_vector, item_vectors)[0]
    
    similarity = [(item, sim) for item, sim in zip(interactions_matrix.columns, sim_scores) if item != item_id]
    
    similarity.sort(key=lambda x: x[1], reverse=True)
    
    most_similar_items = [tup[0] for tup in similarity]
    sim_scores = [tup[1] for tup in similarity]
    
    return most_similar_items, sim_scores


# Collaborative Filtering (item-based)
def item_based_recommendations(user_index, num_of_products, interactions_matrix):
    user_interactions = interactions_matrix.iloc[user_index]
    interacted_items = set(user_interactions[user_interactions > 0].index)
    
    similarity_list = []
    for item_id in interacted_items:
        similar_items_list, sim_scores = similar_items(item_id, interactions_matrix)
        for item, score in zip(similar_items_list, sim_scores):
            if item not in interacted_items:
                similarity_list.append((item, score))
    
    similarity_list.sort(key=lambda x: x[1], reverse=True)
    
    recommendations = [item for item, score in similarity_list[:num_of_products]]
    return recommendations

###################################################

def read_jsonl(filename):
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return pd.DataFrame(data_list)


df1 = read_jsonl('data/Handmade.jsonl')
df2 = read_jsonl('data/meta_Handmade.jsonl')
df3 = read_jsonl('data/Digital_Music.jsonl')
df4 = read_jsonl('data/meta_Digital_Music.jsonl')


df1 = df1.drop(labels=['title', 'text', 'images', 'timestamp', 'helpful_vote', 'verified_purchase'], axis=1)
df3 = df3.drop(labels=['title', 'text', 'images', 'timestamp', 'helpful_vote', 'verified_purchase'], axis=1)
df2 = df2.drop(
    labels=['features', 'description', 'price', 'videos', 'store', 'categories', 'details', 'bought_together'], axis=1)
df4 = df4.drop(
    labels=['features', 'description', 'price', 'videos', 'store', 'categories', 'details', 'bought_together'], axis=1)

# Merge
merged2_df = pd.merge(df1, df2, on='parent_asin', how='inner', suffixes=('', ''))

merged1_df = pd.merge(df3, df4, on='parent_asin', how='inner', suffixes=('', ''))

merged_df = pd.concat([merged2_df, merged1_df])

merged_df=merged_df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='first')

rows, columns = merged_df.shape
print("No of rows = ", rows)
print("No of columns = ", columns)

# merged_df.isna().sum()
plt.figure(figsize=(10, 5))
merged_df['rating'].value_counts(1).plot(kind='bar')
plt.show()

print('Number of unique USERS in Raw data = ', merged_df['user_id'].nunique())
print('Number of unique ITEMS in Raw data = ', merged_df['parent_asin'].nunique())
most_rated = merged_df.groupby('user_id').size().sort_values(ascending=False)[:10]
#most_rated

# To keep users with more than 18 interactions
counts = merged_df['user_id'].value_counts()
merged_df = merged_df[merged_df['user_id'].isin(counts[counts >= 5].index)]

print('Number of unique USERS in Raw data = ', merged_df['user_id'].nunique())
print('Number of unique ITEMS in Raw data = ', merged_df['parent_asin'].nunique())


merged_df = merged_df.drop(labels=['rating_number', 'average_rating'], axis=1)


# Creating the interaction matrix based on ratings and replacing nan values with 0
final_ratings_matrix = merged_df.pivot_table(index='user_id', columns='parent_asin', values='rating',
                                             aggfunc='mean').fillna(0)

print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)

given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)

possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)

density = (given_num_of_ratings / possible_num_of_ratings)
density *= 100
print('density: {:4.2f}%'.format(density))


final_ratings_matrix['user_index'] = np.arange(0, final_ratings_matrix.shape[0])


final_ratings_matrix.set_index(['user_index'], inplace=True)

# final_ratings_matrix.head()


###### SVD #####
def svd_trun(final_ratings_sparse, final_ratings_matrix,n_components):

    svd = TruncatedSVD(n_components, random_state=42)
    U = svd.fit_transform(final_ratings_sparse)
    sigma = np.diag(svd.singular_values_)
    Vt = svd.components_

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)


    preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)

    #preds_df.head()
    preds_matrix = csr_matrix(preds_df.values)
    
    precision, recall, f1, rmse = calculate_precision_recall_f1(final_ratings_sparse, all_user_predicted_ratings)
    precision_percent = precision * 100
    recall_percent = recall * 100
    f1_percent = f1 * 100

    print(f'Precision: {precision_percent:.2f}%, Recall: {recall_percent:.2f}%, F1 Score: {f1_percent:.2f}%,RMSE: {rmse}')

    return preds_matrix, all_user_predicted_ratings


# def svd(final_ratings_sparse,final_ratings_matrix):
#     U, s, Vt = svds(final_ratings_sparse, k = 250) 

#     # construct diagonal array in SVD
#     sigma = np.diag(s)

#     all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

#     #convert the predicted ratings into a DataFrame
#     preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns=final_ratings_matrix.columns)

#     # Predicted ratings
#     preds_df = pd.DataFrame(abs(all_user_predicted_ratings), columns = final_ratings_matrix.columns)

#     preds_matrix = csr_matrix(preds_df.values)
#     return preds_matrix



### RMSE PERFORMANCE EVALUATION ##

def get_top_k(predictions, k=10):
    top_k = np.argsort(predictions, axis=1)[:, -k:]
    return top_k

def calculate_precision_recall_f1(final_ratings_sparse, predictions, k=10):
    top_k = get_top_k(predictions, k)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for user in range(final_ratings_sparse.shape[0]):
        true_items = final_ratings_sparse[user].indices
        predicted_items = top_k[user]

        true_positives += len(set(predicted_items).intersection(set(true_items)))
        false_positives += len(set(predicted_items) - set(true_items))
        false_negatives += len(set(true_items) - set(predicted_items))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    rmse = np.sqrt(mean_squared_error(final_ratings_sparse.toarray(), predictions))

    return precision, recall, f1, rmse

###################

final_ratings_sparse = csr_matrix(final_ratings_matrix.values)
n_components = 150
preds_matrix, all_user_predicted_ratings = svd_trun(final_ratings_sparse, final_ratings_matrix,n_components)


