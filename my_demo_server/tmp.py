from flask import Flask, request, render_template, redirect, url_for,send_file,make_response
from PIL import Image
import flask
import json
import sys
import pickle
import werkzeug
import os
import time
import subprocess
import sys


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def GetNoteText():
    if request.method == 'POST':
        postData= request
        print("postdata: ",postData)
        files_ids = list(flask.request.files)
        for file_id in files_ids:
            imagefile= flask.request.files[file_id]
            filename=werkzeug.utils.secure_filename(imagefile.filename)
            print("image file:" + imagefile.filename)
            imagefile.save("test1.jpg")
            os.system("python predict.py")
            print("save success")
        
        files_ids = list(flask.request.files)
        print("\nReceived Images : ", files_ids)
        
        print("\nReceived Images[0] : ", flask.request.files)
        res = "response.jpg"
        
        return  send_file(res, mimetype='image/jpeg')
        # return json.dumps({'success':True}), 200, {'ContentType':'application/json'}     
    else:
        # print('QQWWEE1')
        return "GET"
        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
if __name__ == "__main__":
#     app.run(ssl_context='adhoc')
#172.17.0.3
    app.run(host='0.0.0.0', port=3333, debug=True)
    # app.run(host='140.123.102.161', port=9874)