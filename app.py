import os
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
import cv2
import keras
import numpy as np
import urllib.request as urlreq
import matplotlib.pyplot as plt
import shutil
from pylab import rcParams
from keras.models import load_model
from keras import backend as K

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def start_page():
    return render_template('landing.html')

@app.route('/result', methods=['GET', 'POST'])
def upload_file():
	output = False
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part.')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected.')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			uploadFile = request.files['file']
			image = cv2.imdecode(np.fromstring(uploadFile.read(), np.uint8), cv2.IMREAD_UNCHANGED)
			
			apply_model(image)

			output = True
	
	return render_template('landing.html', output = output, init = True)

def apply_model(image):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	x, y, width, depth = 50, 200, 950, 500
	image_cropped = image_rgb
	image_template = image_cropped.copy()
	image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

	haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
	haarcascade = "haarcascade_frontalface_alt2.xml"
	detector = cv2.CascadeClassifier(haarcascade)
	faces = detector.detectMultiScale(image_gray)

	LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
	LBFmodel = "lbfmodel.yaml"
	landmark_detector  = cv2.face.createFacemarkLBF()
	landmark_detector.loadModel(LBFmodel)
	_, landmarks = landmark_detector.fit(image_gray, faces)

	fig = plt.figure()

	for landmark in landmarks:
		for x,y in landmark[0]:
			cv2.circle(image_cropped, (int(x), int(y)), 1, (255, 255, 255), 5)

	plt.axis("off")
	plt.imshow(image_cropped)
	fig.savefig('saved.png')
	shutil.move("saved.png", "static/saved.png")
	
if __name__ == "__main__":
	app.run()


