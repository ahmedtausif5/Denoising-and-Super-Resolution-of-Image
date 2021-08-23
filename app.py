
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from numpy import array, argmax, expand_dims, argsort
from cv2 import resize, cvtColor, COLOR_GRAY2BGRA, COLOR_BGRA2BGR
from PIL import Image, UnidentifiedImageError
import cv2
import matplotlib.pyplot as plt
from cv2 import dnn_superres
import numpy as np
import base64
import io
import copy


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':

        """
        1. Taking image input from front end, (html), in input_image_PIL , type(input_image_PIL) = PIL Image
        2. Converting input PIL Image to RGB (To avoid collision with RGBA or other formats)
        3. Converting input_image_PIL to input_image_ndArray,
        because denoising needs numpy.ndarray datatype for cv2 denoising function to work
        """

        try:
            input_image_PIL = Image.open(request.files['file'])

        except UnidentifiedImageError:
            input_image_PIL = None

        if input_image_PIL is not None:

            input_image_PIL_RGB = input_image_PIL.convert('RGB')
            original_image_PIL_RGB = copy.deepcopy(input_image_PIL_RGB)
            input_image_ndArray = array(input_image_PIL_RGB)

            denoise_filter = request.form.get('denoise_filter')
            superRes_model = request.form.get('superRes')

            if denoise_filter:
                if denoise_filter == 'blur':
                    # denoising image using cv2.blur
                    input_image_ndArray = cv2.blur(input_image_ndArray, (3,3))

                if denoise_filter == 'GaussianBlur':
                    # denoising image using cv2.GaussianBlur
                    input_image_ndArray = cv2.GaussianBlur(input_image_ndArray, (3,3), 0)

                if denoise_filter == 'medianBlur':
                    # denoising image using cv2.medianBlur
                    input_image_ndArray = cv2.medianBlur(input_image_ndArray, 3)

                if denoise_filter == 'bilateralFilter':
                    # denoising image using cv2.bilateralFilter
                    input_image_ndArray = cv2.bilateralFilter(input_image_ndArray, 15, 75, 75)

                if denoise_filter == 'fastNlMeansDenoisingColored':
                    # denoising image using cv2.fastNlMeansDenoisingColored
                    input_image_ndArray = cv2.fastNlMeansDenoisingColored(input_image_ndArray, None, 10, 10, 7, 21)


            if superRes_model:
                # Creating Super Res object
                sr = dnn_superres.DnnSuperResImpl_create()

                if superRes_model == 'FSRCNN_x3':
                    # Reading Super Res model
                    sr.readModel("models/FSRCNN_x3.pb")
                    # Setting the modelname and scale
                    sr.setModel("fsrcnn", 3)
                    # Upscaling the denoised image
                    input_image_ndArray = sr.upsample(input_image_ndArray)

                if superRes_model == 'EDSR_x4':
                    # Reading Super Res model
                    sr.readModel("models/EDSR_x4.pb")
                    # Setting the modelname and scale
                    sr.setModel("edsr", 4)
                    # Upscaling the denoised image
                    input_image_ndArray = sr.upsample(input_image_ndArray)

                if superRes_model == 'ESPCN_x4':
                    # Reading Super Res model
                    sr.readModel("models/ESPCN_x4.pb")
                    # Setting the modelname and scale
                    sr.setModel("espcn", 4)
                    # Upscaling the denoised image
                    input_image_ndArray = sr.upsample(input_image_ndArray)

                if superRes_model == 'LapSRN_x8':
                    # Reading Super Res model
                    sr.readModel("models/LapSRN_x8.pb")
                    # Setting the modelname and scale
                    sr.setModel("lapsrn", 8)
                    # Upscaling the denoised image
                    input_image_ndArray = sr.upsample(input_image_ndArray)


            # Converting the above 'numpy.ndarray' datatype of input_image_ndArray to 'PIL Image' datatype
            input_image_PIL_RGB = Image.fromarray(np.uint8(input_image_ndArray)).convert('RGB')

            # Getting the in-memory info
            data1 = io.BytesIO()
            data2 = io.BytesIO()

            # Saving images as in-memory. (.save function can only be applied to PIL objects)
            original_image_PIL_RGB.save(data1, "JPEG")
            input_image_PIL_RGB.save(data2, "JPEG")

            # Getting shape of original image and final image.
            original_width, original_height = original_image_PIL_RGB.size
            final_width, final_height = input_image_PIL_RGB.size

            # Encoding the saved image files.
            original_image = base64.b64encode(data1.getvalue())
            final_image = base64.b64encode(data2.getvalue())

            if not denoise_filter:
                denoise_filter = "None"
            if not superRes_model:
                superRes_model = "None"

            return render_template('result.html', denoise_filter = denoise_filter, superRes_model = superRes_model,
                                    original_width = original_width, original_height = original_height,
                                    final_width = final_width, final_height = final_height,
                                    original_image = original_image.decode('utf-8'),
                                    final_image = final_image.decode('utf-8')
                                    )
        else:
            return render_template('invalid.html')


if __name__ == '__main__':
    app.run(debug=True)
