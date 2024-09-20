import algo.algorithm as main


def show_interacted_products(user_index):

    interacted_products = main.final_ratings_matrix.loc[user_index]
    interacted_product_ids = interacted_products[interacted_products > 0].index.tolist()
    return_data = []

    for interacted_product_id in interacted_product_ids:
        return_data.append(main.show(interacted_product_id, main.merged_df))

    return return_data

def show_suggestions_func(user_index,num_of_products,selected_algorithm):

    if selected_algorithm == "User-Based Collaborative Filtering":
        suggested_products = main.recommendations(user_index, num_of_products, main.final_ratings_matrix, main.merged_df)
    elif selected_algorithm == "Item-Based Collaborative Filtering":
         suggested_products = main.item_based_recommendations(user_index, num_of_products, main.final_ratings_matrix)
    elif selected_algorithm == "SVD":
        suggested_products = main.recommend_items(user_index, main.final_ratings_sparse, main.preds_matrix, main.final_ratings_matrix,num_of_products, main.merged_df)
    
    return_data = []
    for product in suggested_products:
        return_data.append(main.show(product, main.merged_df))

    return return_data

