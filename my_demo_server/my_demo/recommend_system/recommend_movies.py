import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# from defaults import _C as cfg
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, FiBiNET, xDeepFM, ONN, AutoInt
from .train_recommend_movies import get_train_fixlen_feature_columns, get_train_LabelEncoder_info

def data_preprocess(data_df):
    data = data_df.copy(deep=True)
    sparse_features = ["movie_id", "gender", "age"]
    # movie_genres = [
    #     'Action','Adventure','Animation','Childrens','Comedy','Crime',
    #     'Documentary','Drama','Fantasy','Film_Noir','Horror','Musical',
    #     'Mystery','Romance','Sci_Fi','Thriller','War','Western'
    #     ]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # TODO: Add arguments
    encoder_list = get_train_LabelEncoder_info('./recommend_data/data.csv')
    for feat in sparse_features:
        
        # print(feat)
        # print(list(encoder_list[feat].classes_))
        # The range of age is from 1-100
        if feat == 'age':
            age_encoder = np.array([str(i) for i in range(1,101) ])
            encoder_list[feat].classes_ = age_encoder

        elif feat == 'gender':
            gender_encoder = np.array(['M', 'F'])
            encoder_list[feat].classes_ = gender_encoder

        data[feat] = encoder_list[feat].transform(data[feat])

    # 2.count #unique features for each sparse field
    # TODO: Add arguments
    fixlen_feature_columns = get_train_fixlen_feature_columns('./recommend_data/data.csv')
    
    # 他自己的資料型態
    # SparseFeat(name='movie_id', vocabulary_size=187, embedding_dim=4, use_hash=False, dtype='int32', embedding_name='movie_id', group_name='default_group')
    linear_feature_columns = fixlen_feature_columns  
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)  # movie_id, gender, age.

    test_model_input = {name: data[name] for name in feature_names}  # dict of movie_id, gender, age value

    return test_model_input, linear_feature_columns, dnn_feature_columns

def recommend_movies(pretrained, inputs, linear_feature_columns, dnn_feature_columns, DEVICE, df):
    data = df
    # print(data)
    # model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=DEVICE)
    model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='regression', device=DEVICE)
    model.load_state_dict(torch.load(pretrained))
    pred_ans = model.predict(inputs, batch_size=256)

    # print(f'Predict rating: {pred_ans}')

    pred_movie_list = []
    pred_movie_genres = []
    pred_rating = []
    idx = np.argsort(pred_ans, axis=0)[::-1]

    movie_genres = [ col for col in data.columns if col not in ['movie_id','movie_title', 'unknown'] ]

    for i, ans_idx in enumerate(idx[:,0]):
        # TODO: Add arguments (rank)
        if i < 2:
            genres = [ movie_genre for movie_genre in movie_genres 
                        if data.iloc[ans_idx][movie_genre] == 1
            ]
            print(f"Predict rating: {pred_ans[ans_idx][0] :.3f}, movie_id: {data.iloc[ans_idx]['movie_id']}, movie_title: {data.iloc[ans_idx]['movie_title']}, gender: {data.iloc[ans_idx]['gender']}, age: {data.iloc[ans_idx]['age']}, genres: {genres}")
            pred_movie_list.append(data.iloc[ans_idx]['movie_title'])
            pred_movie_genres.append(genres)
            pred_rating.append(pred_ans[ans_idx][0])
            # print(inputs['movie_id'].iloc[i])

    
    # print('movie_max',data.loc[:, ['movie_id']].max(axis=0)) # 1682

    return pred_movie_list, pred_movie_genres, pred_rating

if __name__ == "__main__":
    # test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess(pd.read_csv(cfg.PATH.RECOMMEND_DATA))
    test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess(pd.read_csv("./recommend_data/test.csv"))

    # test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess(pd.read_csv("./recommend_data/data.csv"))
    # pred_movie_list, pred_rating = recommend_movies('./recommend_system/save_model/xDeepFM_MSE1.0181.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0', pd.read_csv("./recommend_data/test.csv"))
    pred_movie_list, pred_rating = recommend_movies('./recommend_system/save_model/FiBiNET_MSE1.0043.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0', pd.read_csv("./recommend_data/test.csv"))
    print(type(a))
    print(a)