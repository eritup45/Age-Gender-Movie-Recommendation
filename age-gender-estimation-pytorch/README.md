# age gender estimation

## Description
Estimate age and gender.

## Demo
python my_demo.py --resume [./checkpoint/xxx.pth]

## Age and Gender Execution
Train:
```bash
python train.py --data_dir appa-real-release
```
Test:
```bash
python test.py --data_dir appa-real-release --resume [./checkpoint/xxx.pth]
```
## Dataset
4113 train, 1500 valid and 1978 test
The apparent age ratings are provided in the files merge_avg_train.csv, merge_avg_test.csv and merge_avg_valid.csv.
Download and extract the [APPA-REAL dataset](https://drive.google.com/drive/folders/1u6s8yQCzcBdstuo6gr14x6fGr0epFpuY?usp=sharing).

## Pretrained model
https://drive.google.com/drive/folders/1boY0wbpossh-I0gG3gx7ZSZp_msHIsT0?usp=sharing

## Reference
https://github.com/yu4u/age-estimation-pytorch
