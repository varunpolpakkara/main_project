from flask import Flask, render_template, request, redirect, url_for, make_response, send_file
from werkzeug import secure_filename
import os
import sys
from functools import wraps, update_wrapper
from datetime import datetime

sys.path.insert(0, '/home/varun/Documents/project/LicensePlateRecognition')

import Extraction
import Main
import test

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/imageOg', methods=['GET', 'POST'])
@nocache
def imageOg():
    return send_file('static/outputs/imgOriginalScene.jpg')

@app.route('/logo', methods=['GET', 'POST'])
@nocache
def imageLogo():
    return send_file('static/outputs/logo1.jpg')

@app.route('/index', methods=['GET', 'POST'])
@nocache
def index():
    return render_template('frontpage.html')

@app.route('/output', methods=['GET', 'POST'])
@nocache
def output():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('index')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect('index')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            fullname = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            file.save(fullname)

            Extraction.init(fullname)
            text = Main.main()
            logotext = test.testing()

            return render_template('display.html', text=text, logotext=logotext)
    else:
        return redirect('index')
