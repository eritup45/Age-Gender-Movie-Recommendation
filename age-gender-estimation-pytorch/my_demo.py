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

def print_time(func):
    import datetime
    def wrap():
        starttime = time.time()
        func()
        endtime = time.time()
        print('Executed Time:', (endtime - starttime), 'sec')
    return wrap

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

def draw_text(img, point, text, drawType="custom"):
    '''
        Be used to draw rectangle and label
    '''
    fontScale = 0.4
    thickness = 5
    text_thickness = 1
    bg_color = (255, 0, 0)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 255, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, '%d' % (text), point, fontFace, 0.5, (255, 0, 0))
    return img
 
 
def draw_label(img, point, label: str, drawType="custom"):
    """
        Be used to draw rectangle and label with '\n'
    """
    fontScale = 0.4
    thickness = 5
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    label = label.split("\n")
    text_size, baseline = cv2.getTextSize(str(label), fontFace, fontScale, thickness)
    for i, text in enumerate(label):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, drawType)
    return img

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
        print(img_path)
        img = cv2.imread(str(img_path), 1)

        if img is not None:
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r))), img_path.name

# one sec one picture
def produce_data(ages, gender):
    # index,user_id,movie_id,rating,timestamp,age,gender,occupation,zip
    """
        Produce all movies id by people's age and gender  
        params:
            gender: 0 or 1 values output by age-gender-model
    """
    age = int(round(ages))
    ML = pd.read_csv(cfg.PATH.MOVIE_LIST)
    movie_len = len(ML)
    data = []
    rand_movie = random.sample(range(movie_len), cfg.DEMO.PRODUCED_MOVIES)
    for i in rand_movie:
    # for i in range(0, movie_len):
        gender_out = 'M' if gender == 0 else 'F'
        d = {'movie_id': ML.iloc[i]['movie_id'], 'age': f'{age}', 'gender': gender_out, 'movie_title': ML.iloc[i]['movie_title']}
        data.append(d)
    data_df = pd.DataFrame(data)
    return data_df

@print_time
def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    if args.output_dir is not None:
        if args.img_dir is None:
            raise ValueError("=> --img_dir argument is required if --output_dir is used")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    # Load 作者model
    if resume_path is None:
        resume_path = Path(__file__).resolve().parent.joinpath("misc", "epoch044_0.02343_3.9984.pth")

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    model.eval()
    margin = args.margin
    img_dir = args.img_dir
    detector = dlib.get_frontal_face_detector()
    img_size = cfg.MODEL.IMG_SIZE
    image_generator = yield_images_from_dir(img_dir) if img_dir else yield_images()

    starttime_without = time.time() ### delete

    with torch.no_grad():
        for img, name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)

                starttime = time.time() ### delete

                age_out, gender_out = model(inputs)
                
                ### delete
                endtime = time.time()
                print('Only age model. Executed Time:', (endtime - starttime), 'sec')

                age_prob = F.softmax(age_out, dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (age_prob * ages).sum(axis=-1)
                predicted_gender = gender_out.max(1)[1]
                print(f'predicted_ages:{predicted_ages}, predicted_gender:{predicted_gender}')

                for i, (d, ages, gender) in enumerate(zip(detected, predicted_ages, predicted_gender)):
                    starttime = time.time() ### delete

                    df = produce_data(ages, gender)
                    # print(df)

                    ### delete
                    endtime = time.time()
                    print('produce_data. Executed Time:', (endtime - starttime), 'sec')                    
                    starttime = time.time() ### delete
                    
                    test_model_input, linear_feature_columns, dnn_feature_columns = data_preprocess(df)
                    # FIXED!!!! 因為df維度與train不一樣，所以爆炸 (ex. df只有男 => 爆炸)
                    pred_movie_list, pred_rating = recommend_movies('./recommend_system/save_model/xDeepFM_MSE1.0181.h5', test_model_input, linear_feature_columns, dnn_feature_columns, 'cuda:0', df)
                    
                    ### delete
                    endtime = time.time()
                    print('recommend model. Executed Time:', (endtime - starttime), 'sec')                    
                    
                    gender_out = 'M' if gender == 0 else 'F'
                    print(f'age_out: {int(round(ages))}, gender_out: {gender_out}\n, pred_movie_list: {pred_movie_list}')
                    
                    label = f"{int(round(ages))}, {gender_out}, {pred_rating[0] :.2f}\n{pred_movie_list[0]}" 
                    # draw_label(img, (d.left(), d.top()), label)
                    draw_label(img, (d.left(), d.bottom()), label)

            if args.output_dir is not None:
                output_path = output_dir.joinpath(name)
                cv2.imwrite(str(output_path), img)
            else:
                cv2.imshow("result", img)
                key = cv2.waitKey(-1) if img_dir else cv2.waitKey(30)

                if key == 27:  # ESC
                    break

    ### delete
    endtime_without = time.time()
    print('Without loading model. Executed Time:', (endtime_without - starttime_without), 'sec')   

if __name__ == '__main__':
    main()
