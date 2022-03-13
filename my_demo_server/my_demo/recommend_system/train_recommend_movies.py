import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM, FiBiNET, xDeepFM, ONN, AutoInt

import numpy as np

def get_train_fixlen_feature_columns(csv_file):
    data = pd.read_csv(csv_file)
    sparse_features = ["movie_id", "gender", "age"]
    movie_genres = [
        'Action','Adventure','Animation','Childrens','Comedy','Crime',
        'Documentary','Drama','Fantasy','Film_Noir','Horror','Musical',
        'Mystery','Romance','Sci_Fi','Thriller','War','Western'
        ]
    sparse_features.extend(movie_genres)
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features]
    return fixlen_feature_columns

# TODO movie_genres要加進去?
def get_train_LabelEncoder_info(csv_file):
    data = pd.read_csv(csv_file)
    sparse_features = ["movie_id", "gender", "age"]
    encoder_list = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        encoder_list.update({feat: lbe.fit(data[feat])})
    return encoder_list

def train_recommend_movies(csv_file, DEVICE):
    """
        Description:
            Train recommend system on: 
                Model: "xDeepFM", 
                Target: "rating",
                Input features: ["movie_id", "gender", "age"],
                Save model to: "save_model/xDeepFM_MSE{}.h5"

        Parameters: 
            csv_file: "path to *.csv"
            DEVICE: "cuda:0"
    """
    data = pd.read_csv(csv_file)
    # sparse_features = ["movie_id", "user_id",
    #                    "gender", "age", "occupation", "zip"]
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
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)  # movie_id, user_id, gender, age, occupation, zip.

    # 3.generate input data for model
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}  # dict of movie_id, user_id, gender, age, occupation, zip values
    
    # 4.Define Model,train,predict and evaluate
    # model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='regression', device=DEVICE)
    # model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
    pred_ans = model.predict(test_model_input, batch_size=256)

    print("test MSE", round(mean_squared_error(test[target].values, pred_ans), 4))
    print("test MAE", round(mean_absolute_error(test[target].values, pred_ans), 4))

    # torch.save(model.state_dict(), './recommend_system/save_model/xDeepFM_MSE{}.h5' .format(round(mean_squared_error(test[target].values, pred_ans), 4)))
    torch.save(model.state_dict(), './recommend_system/save_model/FiBiNET_MSE{}.h5' .format(round(mean_squared_error(test[target].values, pred_ans), 4)))

if __name__ == "__main__":
    train_recommend_movies("./recommend_data/data.csv", 'cuda:0')