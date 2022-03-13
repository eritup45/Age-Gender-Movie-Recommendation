from yacs.config import CfgNode as CN

_C = CN()

# Model
_C.MODEL = CN()
# _C.MODEL.ARCH = "se_resnext50_32x4d"  # check python train.py -h for available models
_C.MODEL.ARCH = ""
_C.MODEL.IMG_SIZE = 224

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "adam"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 80
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
_C.TEST.BATCH_SIZE = 64

# Path
_C.PATH = CN()
_C.PATH.RECOMMEND_DATA = "./recommend_data/data.csv"
_C.PATH.MOVIE_LIST = "./recommend_data/movie_list.csv"

# Demo
_C.DEMO = CN()
_C.DEMO.PRODUCED_MOVIES = 200
_C.DEMO.URL_AGE = "http://140.123.105.233:3334/age"
_C.DEMO.URL_LATEST_RECOMMENDATION = "http://140.123.105.233:3334/latest_recommendation"

