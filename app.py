from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from numpy import array, argmax, expand_dims, argsort
# import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from os import listdir
from tensorflow.keras.models import load_model
from cv2 import resize, cvtColor, COLOR_GRAY2BGRA, COLOR_BGRA2BGR
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
       denoise_kernel = request.form.get('denoise_kernel')
       superRes_model = request.form.get('superRes')

       # data = request.files['file']
       img = Image.open(request.files['file'])
       img = array(img)

       denoised_image = cv2.bilateralFilter(img, 15, 75, 75)
       x = type(denoised_image)
       cv2.imshow("Input Image", img)
       cv2.imshow("Final Image", denoised_image)
       cv2.waitKey(0)
       cv2.destroyAllWindows()




    return render_template('result.html', denoise_kernel=denoise_kernel,superRes_model=superRes_model, x=x)


if __name__ == '__main__':
    app.run(debug=True)
