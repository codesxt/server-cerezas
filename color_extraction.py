from skimage import color
from skimage import io
from skimage.util.dtype import dtype_range
from skimage import exposure
from skimage.color import rgb2lab, lab2lch

import cv2

colors = []
colors.append((42.074, 67.486, 41.053))
colors.append((35.522, 55.752, 45.689))
colors.append((30.286, 49.517, 30.363))
colors.append((24.539, 32.595, 17.289))
colors.append((19.291, 19.48, 0.942))
colors.append((11.7956, -0.0078, 1.5308))

def distancia_color(imagen_lab, color_id):
  return color.deltaE_ciede2000(imagen_lab,colors[color_id])

def analisis_distancia(matriz,dimensiones,largo):
  elementos = 0
  for j in range(0,largo):
    for x in range(0,dimensiones):
      elementos = elementos + matriz[j][x]
  elementos = elementos / (largo * dimensiones)
  return elementos

def similitud_color(imagen):
  imagen_lab = rgb2lab(imagen)
  dimensiones = len(imagen[0])
  distancia = []
  for i in range(len(colors)):
    matriz = distancia_color(imagen_lab, i)
    largo = len(matriz)
    distancia.append(analisis_distancia(matriz,dimensiones,largo))
  return distancia

def establecer_color(distancia):
  deltaE    = 100
  resultado = -1
  for i in range(0,6):
    if distancia[i] < deltaE :
      deltaE = distancia[i]
  for i in range(0,6):
    if distancia[i] == deltaE:
      resultado = i
  return resultado

def getClassFromURL(url):
  imagen = io.imread(url)
  distancias = similitud_color(imagen)
  clase = establecer_color(distancias)
  return clase

def getClassFromImage(image):
  distancias = similitud_color(image)
  clase = establecer_color(distancias)
  return clase

if __name__ == "__main__":
    clase = getClassFromURL("1.jpg")
    print("La imagen corresponde a la clase ", clase+1)
