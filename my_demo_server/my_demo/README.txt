## Demo
python my_demo.py --resume [./checkpoint/xxx.pth]

## Execute
Train:
    python train.py --data_dir appa-real-release

Test:
    python test.py --data_dir appa-real-release --resume [./checkpoint/xxx.pth]

## Dataset
4113 train, 1500 valid and 1978 test
The apparent age ratings are provided in the files merge_avg_train.csv, merge_avg_test.csv and merge_avg_valid.csv.

## LOGS

7/26
將資料夾test改成backup_test
將merge_avg_test改成backup_merge_avg_test

上傳自己照片test1, test2
上傳merge_avg_test.csv

注意: 到時須將dataset.py的67行"_face.jpg"移除

