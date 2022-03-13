import argparse
import better_exceptions
from pathlib import Path
from contextlib import contextmanager
import urllib.request
import numpy as np
import cv2
import dlib
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import random
from model import get_model
from defaults import _C as cfg
from recommend_system.recommend_movies import data_preprocess, recommend_movies
import datetime
import time
import requests
import json

def get_args():
    parser = argparse.ArgumentParser(description="Age estimation demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=str, default=None,
                        help="Model weight to be tested")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="Margin around detected face for age-gender estimation")
    parser.add_argument("--img_dir", type=str, default='./my_in',
    # parser.add_argument("--img_dir", type=str, default=None,
                        help="Target image directory; if set, images in image_dir are used instead of webcam")
    parser.add_argument("--output_dir", type=str, default='./my_out',
    # parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory to which resulting images will be stored if set")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


def yield_images():
    with video_capture(0) as cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img, None


def yield_images_from_dir(img_dir):
    img_dir = Path(img_dir)

    for img_path in img_dir.glob("*.*"):
        print('img_path:', img_path) # delete
        img = cv2.imread(str(img_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), img_path.name

    
def main():
    
    args = get_args()

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = args.img_dir
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()
    
    starttime_without = time.time() ### delete

    count = 1
    frames = 24
    for img, name in image_generator:
        if count / frames != 1:
            count += 1
            continue
        else:
            count = 1

        starttime_one = time.time() ### delete
        starttime = time.time() ### delete

        # Client should pass 'img' in server
        client_data = img.tolist()
        client_data = {'img': client_data}
        r = requests.request("POST", cfg.DEMO.URL_AGE, json=client_data)

        ### delete
        endtime = time.time()
        print('One image all done. Executed Time:', (endtime - starttime), 'sec')

        ans = json.loads(r.text)
        # Server return processed picture.
        img = np.uint8(np.array(ans['img']))


        # print('---------------------')
        # print(r.text)
        # print('---------------------')
        # print(ans)
        # print(np.array(ans['img']))
        # print(np.array(ans['img']).shape)

        if args.output_dir is not None:
            output_path = output_dir.joinpath(name)
            cv2.imwrite(str(output_path), img)
        else:
            cv2.imshow("result", img)
            key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

            if key == 27:  # ESC
                break

        ### delete
        endtime_one = time.time()
        print('One image all done. Executed Time:', (endtime_one - starttime_one), 'sec')

    ### delete
    endtime_without = time.time()
    print('Without loading model. Executed Time:', (endtime_without - starttime_without), 'sec')   

if __name__ == '__main__':
    main()
