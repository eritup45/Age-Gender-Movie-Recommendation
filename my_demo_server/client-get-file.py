import requests
import sys
import cv2
import torch

# TODO: torch需先轉成numpy，再轉成一個list (一張臉)，用files = {'第幾張臉': [臉]}

class SendFile():
    def __init__(self, s):
        self.s = s

    def sendImg(self, jpgpath, jpgname='banana.jpg', jpgtype='image/jpeg'):
        # 登入並更新cookies
        f = open(jpgname, 'rb')  # 絕對路徑
        url2 = "140.123.105.233:3333"
        body = {
            'localurl': (None, jpgname),
            'imgFile': (jpgname, open(jpgpath, 'rb'), jpgtype)
            # 1、絕對路徑  2、open(jpgname, 'rb')  3、content-type的值
            }
        # 上傳圖片的時候，不data和json，用files
        r = self.s.post(url2, files=body)    # 1、呼叫全域性的s，用self.s   2、files
        print(r.text)
        # 上傳到伺服器，每傳一次地址都不一樣

        # 解決拋異常
        try:
            jpg_url = r.json()['url']   # （相對路徑）
            print(jpg_url)
            return jpg_url

        except Exception as msg:    # 返回報錯資訊
            print('圖片上傳失敗，原因：%s'%msg)   # 列印報錯資訊
            return ''


def client_post():
    SERVER_IP = "140.123.105.233"
    API_SERVER = "http://" + SERVER_IP + ":3333"
    DOWNLOAD_IMAGE_API = "/"
    url = API_SERVER + DOWNLOAD_IMAGE_API

    # myfiles = {'file0': ( (cv2.imread('banana.jpg')) )}
    # x = requests.post(url, files = myfiles)

    # data = {'file1': [cv2.imread('banana.jpg') for i in range(2)]}
    # x = requests.request("POST", url, data=data)

    # OK !!!! (JSON)
    data = {'arr': ( [cv2.imread('banana.jpg').tolist() for i in range(2)])}
    x = requests.request("POST", url, json=data)

    print('This is error output', file=sys.stderr)
    print('This is standard output', file=sys.stdout)

if __name__ == '__main__':
        client_post()