import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, FiBiNET, xDeepFM, ONN, AutoInt

data = pd.read_csv("./recommend_data/data.csv")

def data_preprocess(csv_file):
    data = pd.read_csv(csv_file)
    sparse_features = ["movie_id", "gender", "age"]
    movie_genres = [
        'Action','Adventure','Animation','Childrens','Comedy','Crime',
        'Documentary','Drama','Fantasy','Film_Noir','Horror','Musical',
        'Mystery','Romance','Sci_Fi','Thriller','War','Western'
        ]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    
    sparse_features.extend(movie_genres)
    # 2.count #unique features for each sparse field
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    
    # 他自己的資料型態
    # SparseFeat(name='movie_id', vocabulary_size=187, embedding_dim=4, use_hash=False, dtype='int32', embedding_name='movie_id', group_name='default_group')
    linear_feature_columns = fixlen_feature_columns  
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)  # movie_id, gender, age.

    test_model_input = {name: data[name] for name in feature_names}  # dict of movie_id, gender, age value

    return test_model_input, linear_feature_columns, dnn_feature_columns

def test_recommend_movies(pretrained, inputs, linear_feature_columns, dnn_feature_columns, DEVICE):
    # model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=DEVICE)
    model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='regression', device=DEVICE)
    model.load_state_dict(torch.load(pretrained))
    pred_ans = model.predict(inputs, batch_size=256)

    print(f'Predict rating: {pred_ans}')

    pred_movie_list = []
    idx = np.argsort(pred_ans, axis=0)[::-1]
    for i, ans_idx in enumerate(idx[:,0]):
        if i < 5:
            print(f"Predict rating: {pred_ans[ans_idx][0] :.3f}, movie_id: {data.iloc[ans_idx]['movie_id']}, gender: {data.iloc[ans_idx]['gender']}, age: {data.iloc[ans_idx]['age']}, user_id: {data.iloc[ans_idx]['user_id']}")
            pred_movie_list.append(data.iloc[ans_idx]['movie_id'])
            # print(inputs['movie_id'].iloc[i])

    
    # print('movie_max',data.loc[:, ['movie_id']].max(axis=0)) # 1682

    return pred_movie_list

if __name__ == "__main__":
    test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess("./recommend_data/data.csv")
    a = test_recommend_movies('./recommend_system/save_model/FiBiNET_MSE1.0043.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0')
    print(type(a))
    print(a)