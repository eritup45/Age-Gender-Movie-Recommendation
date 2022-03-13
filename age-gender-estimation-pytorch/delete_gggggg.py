from recommend_system.recommend_movies import data_preprocess, recommend_movies
import numpy as np
import pandas as pd

df = pd.read_csv("./recommend_data/test.csv")
test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess(pd.read_csv("./recommend_data/test.csv"))
pred_movie_list, pred_rating = recommend_movies('./recommend_system/save_model/FiBiNET_MSE1.0043.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0', df)

print(pred_movie_list, pred_rating)
print('----------')