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
import multiprocessing as mp

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

def send_server(img, name, number):
    args = get_args()
    starttime = time.time() ### delete

    # 壓縮
    # cv2.imwrite("not_compression.jpg", img)
    cv2.imwrite(f"raw_frame{number}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY,30])
    files = {'img': (f"raw_frame{number}.jpg", open(f"raw_frame{number}.jpg", 'rb'), 'image/jpg')}
    r = requests.post(cfg.DEMO.URL_AGE, files=files)
    # print(r.content)

    # Read bytes from server and change to numpy
    filestr = r.content
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    print(img)

    ### delete
    endtime = time.time()
    print('One image all done. Executed Time:', (endtime - starttime), 'sec')

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir.joinpath(name)
        cv2.imwrite(str(output_path), img)  
    else:
        print(img.shape)
        print(img)
        cv2.imshow("result", img)

    return img

def main():
    # multiprocessing
    mp.set_start_method('spawn')
    pool = mp.Pool(processes=2)

    args = get_args()

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = args.img_dir
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()

    count = 1
    frames = 1
    imgs = []
    names = []
    imgs_count = 0
    mp_core = 2
    for img, name in image_generator:
    # while True:
        # TODO: fix to parameters or default.py
        # imgs, names = [next(image_generator) for i in range(mp_core)]
        print('imgs_count', imgs_count)
        if count % frames != 0:
            count += 1
            continue
        else:
            count = 1
            imgs_count += 1
            imgs.append(img)
            names.append(name)

        # TODO: 若用資料夾demo只有單數張會出問題(不會執行)
        # When images are enough to execute the multi process
        if imgs_count % mp_core  != 0:
            continue
        else:
            imgs_count = 0

        img = pool.starmap(send_server, [(imgs[i], names[i], i) for i in range(-mp_core, 0)])

        print(type(img))
        print(len(img))
        print(type(img[1]))
        print((img[1].shape))

        imgs = []
        names = []

        if args.output_dir is not None:
            pass
            # output_path = output_dir.joinpath(name)
            # cv2.imwrite(str(output_path), img)
        else:
            pass
            # cv2.imshow("result", img)
            # key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

            # if key == 27:  # ESC
            #     pool.terminate()
            #     pool.join()
            #     break
        key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)
        if key == 27:  # ESC
            pool.terminate()
            pool.join()
            break

if __name__ == '__main__':
    main()
