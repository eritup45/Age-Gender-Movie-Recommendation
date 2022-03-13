import argparse
import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from sklearn.preprocessing import LabelEncoder


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
#         csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        csv_path = Path(data_dir).joinpath(f"merge_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y_age = []
        self.y_gender = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        # df['gender'] = self.preprocess_gender(df['gender'])
        
        ignore_path = Path(__file__).resolve().parent.joinpath("ignore_list.csv")
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            # TODO: retard preprocess
            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y_age.append(row["apparent_age_avg"])
            self.y_gender.append(self.preprocess_gender(row["gender"]))
            self.std.append(row["apparent_age_std"])

    # Encode            
    def preprocess_gender(self, gender):
        if gender.lower() == 'male':
            return 0
        elif gender.lower() == 'female':
            return 1
        elif gender.lower() == 'm':
            return 0
        elif gender.lower() == 'f':
            return 1

    def __len__(self):
        return len(self.y_age)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y_age[idx]
        gender = self.y_gender[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 100), gender


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = FaceDataset(args.data_dir, "train")
    print("train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "valid")
    print("valid dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, "test")
    print("test dataset len: {}".format(len(dataset)))

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                              num_workers=2, drop_last=True)
    for i, (img, age, gender) in enumerate(loader):
        if i == 0:
            print('[test] First person:', age, gender)

if __name__ == '__main__':
    main()
