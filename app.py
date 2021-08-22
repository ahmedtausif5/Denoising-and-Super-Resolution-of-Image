from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from numpy import array, argmax, expand_dims, argsort
from cv2 import resize, cvtColor, COLOR_GRAY2BGRA, COLOR_BGRA2BGR
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres
import numpy as np
import base64
import io


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
       """
       Taking image input from front end, (html), in input_image_PIL , type(input_image_PIL) = PIL Image
       Converting input_image_PIL to input_image_ndArray,
       because denoising needs numpy.ndarray datatype for cv2 denoising function to work
       """
       input_image_PIL = Image.open(request.files['file'])
       input_image_ndArray = array(input_image_PIL)

       # denoising image using cv2
       denoised_image_ndArray = cv2.bilateralFilter(input_image_ndArray, 15, 75, 75)

       # converting the above 'numpy.ndarray' datatype of denoised_image_ndArray to 'PIL Image' datatype
       denoised_image_PIL = Image.fromarray(np.uint8(denoised_image_ndArray)).convert('RGB')

       # Get the in-memory info
       data1 = io.BytesIO()
       data2 = io.BytesIO()

       # Saving images as in-memory.
       input_image_PIL.save(data1, "JPEG")
       denoised_image_PIL.save(data2, "JPEG")

       # Then encoding the saved image files.
       original_image = base64.b64encode(data1.getvalue())
       final_image = base64.b64encode(data2.getvalue())


    return render_template('result.html', denoise_kernel=denoise_kernel, superRes_model=superRes_model,
                            original_image=original_image.decode('utf-8'), final_image= final_image.decode('utf-8'))


if __name__ == '__main__':
    app.run(debug=True)
