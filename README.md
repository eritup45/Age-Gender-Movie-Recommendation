
# Age Gender Movies Recommendation

## Introduction
Detect people's age and gender, and use them to recommend users movies.

## Dataset
* APPA-REAL (real and apparent age)
    * https://chalearnlap.cvc.uab.cat/dataset/26/description/
* MovieLens 100K Dataset
    * https://www.kaggle.com/prajitdatta/movielens-100k-dataset

## Folder structure
```
    ./Age-Gender-Movie-Recommendation
        +-- age-gender-estimation-pytorch
            +-- appa-real-release
        +-- Movies_Recommendation_DeepCTR 
            +-- data.csv
        +-- my_demo_server 
```
* Age-Gender-Movie-Recommendation
    * Train, test and demo.
* my_demo_server
    * Complete demo. (Including server, client, GUI)

## age-gender-estimation-pytorch

### Description
Estimate age and gender.

### Dataset
4113 train, 1500 valid and 1978 test
The apparent age ratings are provided in the files merge_avg_train.csv, merge_avg_test.csv and merge_avg_valid.csv.
Download and extract the [APPA-REAL dataset](https://drive.google.com/drive/folders/1u6s8yQCzcBdstuo6gr14x6fGr0epFpuY?usp=sharing).

### Pretrained model
https://drive.google.com/drive/folders/1boY0wbpossh-I0gG3gx7ZSZp_msHIsT0?usp=sharing

### Demo

```ps=
cd age-gender-estimation-pytorch
# python my_demo.py --resume [./checkpoint/xxx.pth]
python my_demo.py --resume checkpoint/eff_conv1024_512/epoch057_0.02398_4.2099.pth
```

### Train
```ps=
python train.py --data_dir appa-real-release
```

### Test
```ps=
python test.py --data_dir appa-real-release --resume checkpoint/eff_conv1024_512/epoch057_0.02398_4.2099.pth
```


## Movies_Recommendation_DeepCTR
### Description
    Using 'xDeepFM' to estimate movies' predicting rate.

### Dataset
https://drive.google.com/drive/folders/1OiZo6UK49miC9JzwPfzrCx7ZzbEw1RuE?usp=sharing


### Pretrained models
https://drive.google.com/drive/folders/1XFcurgWVCX9bFnictzLfLm928MyQqazZ?usp=sharing

### Train
```
python train_recommend_movies.py
```

### Test
    Note: There should be ./data.csv in the same directory. 
```
python test_recommend_movies.py
```


### How to use?
```
    test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess("./data.csv")
    a = test_recommend_movies('save_model/xDeepFM_MSE1.0184.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0')    
```


## my_demo_server

### Demo

```shell=
cd ./my_demo_server/my_demo

# 1. Open server
python server_my_demo.py --resume checkpoint/eff_conv1024_512/epoch057_0.02398.2099.pth

# 2. Open client
# Get images from my_in (No GUI and Youtube version)
python client_my_demo.py
# Get images from camera
python client_my_demo.py --Use_dir False

# 3. GUI and Youtube version
python client_my_demo_youtube.py
```

