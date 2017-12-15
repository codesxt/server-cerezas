# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:18:22 2017

@author: bruno
"""

from sanic import Sanic
from sanic.response import json, html, text
from sanic.config import Config
from sanic_cors import CORS, cross_origin
Config.KEEP_ALIVE = False

import os
import uuid

import cv2

from matplotlib import pyplot as plt

from img_proc import *
from color_extraction import *

#web_application = open('views/upload.html', 'r').read()
#web_application = open('client/dist/index.html', 'r').read()

app = Sanic(__name__)

CORS(app)

app.static('/output', './output')
app.static('/', './client/dist')

"""
@app.route("/")
async def test(request):
  return html(web_application)
"""

@app.route('/api/v1/check', methods=['GET'])
async def check_handler(request):
  return json(
      {'alive': 'true'},
      headers={'X-Served-By': 'sanic'},
      status=200
  )

@app.route('/api/v1/upload', methods=['POST'])
async def post_handler(request):
  # print("Debugging")
  # print(print(dir(request)))
  # print(request.content_type)
  if request.files.get("image") is None:
    return json(
        {'message': 'Upload data is not valid.', 'error':'IMAGE_NOT_SET'},
        headers={'X-Served-By': 'sanic'},
        status=400
    )
  else:
    # Extrae metadatos del archivo recibido
    image_filename, image_extension = os.path.splitext(request.files.get('image').name)
    image_filetype = request.files.get('image').type
    # Extrae los datos del archivo recibido
    image_data = bytearray(request.files.get('image').body)

    # Crea un ID único para el proceso y guarda el archivo de entrada
    hash = uuid.uuid4().hex
    out_file = open('input/' + hash + image_extension, 'wb')  # Output file
    out_file.write(image_data)
    out_file.close()
    input_file = 'input/' + hash + image_extension
    image_output = 'output/' + hash + image_extension

    # Rectifica la imagen usando un marcador de referencia
    image_rect, pix_scale = rectify_image('marker.png', 4, input_file)
    if image_rect is None:
     return json(
        {'message': 'Image could not be rectified.', 'error': 'NOT_RECTIFIED'},
        headers={'X-Served-By': 'sanic'},
        status=400
     )

    # Retorna un arreglo de cerezas segmentadas como imágenes
    segmented_cherries, segmented_histograms, radius_list = segment_cherries(image_rect, image_output)
    ##print(segmented_cherries[0].shape)
    # Crea una carpeta para guardar las cerezas segmentadas
    os.makedirs('output/'+hash, exist_ok=True)
    filenames      = []
    cherry_classes = []
    caliber        = []
    for (pos, cherry) in enumerate(segmented_cherries):
      # Esta línea reemplaza el color negro de las imágenes segmentadas
      # por blanco, para que se vea mejor
      cherry[np.where((cherry==[0,0,0]).all(axis=2))] = [255,255,255]
      filename = 'output/'+hash+'/'+str(pos)+'.jpg'
      cv2.imwrite(filename,cherry)
      filenames.append(filename)
      caliber.append(radius_list[pos]* 2 * pix_scale);
      cherry_classes.append(getClassFromImage(cv2.cvtColor(cherry,cv2.COLOR_BGR2RGB)))
    if(true):
      directory = "/cerezas"
    else:
      directory = ""

    results = [{"filename": "http://" + request.host + directory + "/" + f, "class": c, "caliber": s} for f, c, s in zip(filenames, cherry_classes, caliber)]
    return json(
        {
          'message'  : 'Image uploaded successfully.',
          'filename' : image_filename,
          'extension': image_extension,
          'type'     : image_filetype,
          'results'  : results,
          'cherries' : len(segmented_cherries),
          'result'   : request.host + directory + '/' + image_output
        },
        headers={'X-Served-By': 'sanic'},
        status=200
    )

app.run(host="0.0.0.0", port=8000)
