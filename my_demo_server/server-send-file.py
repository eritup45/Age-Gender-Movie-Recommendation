import io
from flask import Flask, send_file, request
import sys
import torch
import numpy as np

app = Flask(__name__)
# if request.method == 'POST':
    #     print(request)
        # # get faces from client
        # images = requ
        # age_out, gender_out = model(inputs)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print('\n\n\n\n\nGet Something!!!')

    # print(request.files['file0'])   
    print('\n') 

    # NOT OK !!! (FROM STRING)
    # print(request.form.getlist('file1'))
    # print(request.form['file1'])
    # print(type(request.form['file1']))
    # arr = np.fromstring(request.form['file1']).reshape(2, 3, -1)
    # print(arr)
    # print(arr.shape)
    print('\n') 

    # OK (JSON)!!!!
    data_json = request.json
    arr = np.array(data_json['arr'])
    print(arr)
    print(arr.shape)
    print('\n') 

    print('This is error output', file=sys.stderr)
    print('This is standard output', file=sys.stdout)
    print('\n')
    return 'Hi'

if __name__ == '__main__':
    app.run(host="140.123.105.233", port=3333, debug=True)