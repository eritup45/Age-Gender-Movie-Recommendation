# Movie recommendation

## Description
    Using 'xDeepFM' to estimate movies' predicting rate. 

## Dataset
https://drive.google.com/drive/folders/1OiZo6UK49miC9JzwPfzrCx7ZzbEw1RuE?usp=sharing

## Train
```
python train_recommend_movies.py
```

## Test
    Note: There should be ./data.csv in the same directory. 
```
python test_recommend_movies.py
```

## Pretrained models
https://drive.google.com/drive/folders/1XFcurgWVCX9bFnictzLfLm928MyQqazZ?usp=sharing

## How to use?
```
    test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess("./data.csv")
    a = test_recommend_movies('save_model/xDeepFM_MSE1.0184.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0')    
```

## References
[DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch.git)

