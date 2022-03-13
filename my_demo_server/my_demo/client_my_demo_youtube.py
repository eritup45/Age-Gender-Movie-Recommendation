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
import searchYT
from PyQt5 import QtGui, QtCore, QtWidgets
import PyQt5.QtWebEngineWidgets as QtWebEngineWidgets
from enum import Enum, auto


def get_args():
    parser = argparse.ArgumentParser(description="Age estimation demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=str, default=None,
                        help="Model weight to be tested")
    parser.add_argument("--margin", type=float, default=0.4,
                        help="Margin around detected face for age-gender estimation")
    # parser.add_argument("--img_dir", type=str, default='./my_in',
    parser.add_argument("--img_dir", type=str, default=None,
                        help="Target image directory; if set, images in image_dir are used instead of webcam")
    # parser.add_argument("--output_dir", type=str, default='./my_out',
    parser.add_argument("--output_dir", type=str, default=None,
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
    # 壓縮
    # cv2.imwrite("not_compression.jpg", img)
    cv2.imwrite(f"raw_frame{number}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY,30])
    files = {'img': (f"raw_frame{number}.jpg", open(f"raw_frame{number}.jpg", 'rb'), 'image/jpg')}
    r = requests.post(cfg.DEMO.URL_AGE, files=files)
    movie = requests.get(cfg.DEMO.URL_LATEST_RECOMMENDATION)

    # Read bytes from server and change to numpy
    filestr = r.content
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return img, movie

class MainWidget(QtWidgets.QWidget):

    pressed = QtCore.pyqtSignal(QtGui.QKeyEvent)

    def __init__(self):
        super().__init__()

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        self.pressed.emit(e)

def relativeQUrl(path):
    return QtCore.QUrl.fromLocalFile(QtCore.QDir.currentPath() + path)  

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

    #count = 1
    #frames = 1

    app = QtWidgets.QApplication([])
    widget = MainWidget()
    widget.setWindowTitle('Demo')
    widget.show()
    image_frame = QtWidgets.QLabel()
    image_frame.setFocusPolicy(QtCore.Qt.StrongFocus)
    videoBoxLayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
    videoBoxLayout.setContentsMargins(32, 32, 32, 32)
    videoBoxLayout.addWidget(image_frame)
    videoBox = QtWidgets.QWidget()
    videoBox.setStyleSheet("background-color: #000000");
    videoBox.setLayout(videoBoxLayout)
    layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
    layout.addWidget(videoBox)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    webView = QtWebEngineWidgets.QWebEngineView()
    webView.setMinimumSize(550, 550)
    webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.FocusOnNavigationEnabled, False)
    profile = QtWebEngineWidgets.QWebEngineProfile()
    page = QtWebEngineWidgets.QWebEnginePage(profile, None)
    webView.setPage(page)
    startPageUrl = relativeQUrl("/start_page.html");
    page.load(startPageUrl);
    layout.addWidget(webView);
    widget.setLayout(layout)
    timer = QtCore.QTimer()
    class AppState(Enum):
        IDLE = auto()
        WAIT_RESULT = auto()
        VIEW_RESULT = auto()

    state = AppState.IDLE

    def onWidgetPressed(e):
        nonlocal state
        key = e.key()
        if key == QtCore.Qt.Key_R:
            layout.setDirection(
                (QtWidgets.QBoxLayout.TopToBottom
                if layout.direction() == QtWidgets.QBoxLayout.LeftToRight
                else QtWidgets.QBoxLayout.LeftToRight)
            )
        elif key == QtCore.Qt.Key_Q:
            widget.close()
        elif key == QtCore.Qt.Key_S:
            if state == AppState.IDLE:
                state = AppState.WAIT_RESULT
                img, name = next(image_generator)
                starttime = time.time()
                resultImg, movie = send_server(img, name, -1)
                endtime = time.time()
                app.processEvents()
                print('Result received after', (endtime - starttime), 's')
                timer.stop()
                if args.output_dir is not None:
                    output_dir = Path(args.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir.joinpath(name)
                    cv2.imwrite(str(output_path), img)
                else:
                    showNextFrame(resultImg)
                    print(movie.text)
                    if movie.text != '':
                        page.scripts().clear();
                        script = QtWebEngineWidgets.QWebEngineScript()
                        script.setSourceCode(f"""
const head = document.createElement('head');
document.documentElement.append(head);
const css = document.createElement('style');
css.type = 'text/css';
head.appendChild(css);
css.innerText = `
.style-scope.ytd-masthead, #filter-menu {{
    display: none !important;
}}
#page-manager {{
    margin-top: 0 !important;
}}
ytd-search {{
    padding: 0px 12px !important;
}}
ytd-app {{
    display: none !important;
    background-color: hsla(0, 0%, 90%, 1.0) !important;
}}
div.loading-text {{
    color: hsla(0, 0%, 25%, 1.0);
    font-size: 26px;
}}
div.loading-text > span {{
    color: hsla(0, 0%, 0%, 1.0);
    font-size: 36px;
}}
body, :root {{
    background-color: hsla(0, 0%, 90%, 1.0) !important;
}}
ytd-mini-guide-renderer {{
    display: none !important;
}}
ytd-page-manager {{
    margin-left: 0 !important;
}}
`;
const body = document.createElement('body');
document.documentElement.append(body);
const loadingTextDiv = document.createElement('div');
const movieSpan = document.createElement('span');
body.append(loadingTextDiv);
movieSpan.append(`"{movie.text}"`);
loadingTextDiv.append(
    '推薦電影:',
    document.createElement('br'),
    movieSpan,
    document.createElement('br'),
    document.createElement('br'),
    '載入相關影片...');
loadingTextDiv.classList.toggle('loading-text');
document.addEventListener('DOMContentLoaded', () => {{
    const css = document.createElement('style');
    css.type = 'text/css';
    head.appendChild(css);
    css.innerText = `
    ytd-app {{
        display: block !important;
    }}
    `;
}});

                        """);
                        script.setInjectionPoint(QtWebEngineWidgets.QWebEngineScript.DocumentCreation)
                        page.scripts().insert(script);
                        page.load(QtCore.QUrl(searchYT.get_youtube_search_url(movie.text)))
                        state = AppState.VIEW_RESULT
                    else:
                        print('No face detected')
                        page.scripts().clear();
                        noFaceUrl = relativeQUrl("/no_face.html");
                        page.load(noFaceUrl);
                        timer.start()
                        state = AppState.IDLE
            elif state == AppState.VIEW_RESULT:
                timer.start()
                state = AppState.IDLE

    widget.pressed.connect(onWidgetPressed)
    timer.setInterval(60)

    def showNextFrame(img):
        qImage = QtGui.QImage(
            img.data,
            img.shape[1],
            img.shape[0],
            QtGui.QImage.Format_RGB888
        ).rgbSwapped()
        pixel_map = QtGui.QPixmap.fromImage(qImage)
        image_frame.setPixmap(pixel_map.scaled(pixel_map.width() * 1.5, pixel_map.height() * 1.5))

    timer.timeout.connect(lambda: showNextFrame(next(image_generator)[0]))
    timer.start()
    app.exec_()


if __name__ == '__main__':
    main()
