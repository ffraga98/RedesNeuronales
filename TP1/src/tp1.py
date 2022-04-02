#%% MODULES
# %reset -sf

from herp.derp.functions import *
from herp.derp.exercises import *
import numpy as np
import os
#%% READING BMP FILES
# path = '/home/fernando/Documentos/Archivos FIUBA/1C2022/REDES_NEURONALES/TP/TP1/img'

# #Leo los nombres de las imagenes
# path_images = open(os.path.join(path,'images.txt'), "r")
# images = path_images.read().split('\n')[0:-1]

images = read_images('images.txt')
#%% APRENDIZAJE
W = ley_hebb_gral(images)

#%% 1) VERIFICACION DE APRENDIZAJE

verificar_aprendizaje('images.txt')

#%% 2) VERSIONES ALTERADAS

evolucion_version_alterada('images_ejb.txt')

#%%% 4) MINIMO LOCAL DESDE ORIGINAL

minimo_local_desde_original('images.txt')