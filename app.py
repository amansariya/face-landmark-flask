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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			image = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/uploads/"+filename)
			
			apply_model(image)

			#redirect(url_for('upload_file',filename='saved.png'))
			
			return render_template('home.html')
			#'''
			#<!doctype html>
			#<title>Results</title>
			#<h1>Image contains a - '''+result+'''</h1>
			#<h2>Dominant color is - '''+color_result+'''</h2>
			#<form method=post enctype=multipart/form-data>
			#  <input type=file name=file>
			#  <input type=submit value=Upload>
			#</form>
			#'''
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	  <input type=file name=file>
	  <input type=submit value=Upload>
	</form>
	'''

def apply_model(image):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	x, y, width, depth = 50, 200, 950, 500
	image_cropped = image_rgb
	#[y:(y+depth), x:(x+width)]
	image_template = image_cropped.copy()
	image_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
	haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
	haarcascade = "haarcascade_frontalface_alt2.xml"
	#if (haarcascade in os.listdir(os.curdir)):
    #	print("File exists")
	#else:
    #	# download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
    #	urlreq.urlretrieve(haarcascade_url, haarcascade)
    #	print("File downloaded")

	detector = cv2.CascadeClassifier(haarcascade)
	faces = detector.detectMultiScale(image_gray)
	LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

	# save facial landmark detection model's name as LBFmodel
	LBFmodel = "lbfmodel.yaml"
	#if (LBFmodel in os.listdir(os.curdir)):
    #	print("File exists")
	#else:
	#	# download picture from url and save locally as lbfmodel.yaml, < 54MB
    #	urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    #	print("File downloaded")

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
	


def catOrDog(image):
	'''Determines if the image contains a cat or dog'''
	classifier = load_model('./models/cats_vs_dogs_V1.h5')
	image = cv2.resize(image, (150,150), interpolation = cv2.INTER_AREA)
	image = image.reshape(1,150,150,3)
	res = str(classifier.predict_classes(image, 1, verbose = 0)[0][0])
	print(res)
	print(type(res))
	if res == "0":
		res = "Cat"
	else:
		res = "Dog"
	K.clear_session()
	return res

def getDominantColor(image):
	'''returns the dominate color among Blue, Green and Reds in the image '''
	B, G, R = cv2.split(image)
	B, G, R = np.sum(B), np.sum(G), np.sum(R)
	color_sums = [B,G,R]
	color_values = {"0": "Blue", "1":"Green", "2": "Red"}
	return color_values[str(np.argmax(color_sums))]
	
if __name__ == "__main__":
	app.run()
	#app.run(host= '0.0.0.0', port=80)


