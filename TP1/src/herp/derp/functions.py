#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 05:50:47 2022

@author: fernando
"""
import numpy as np
import os
import imageio as iio
import matplotlib.pyplot as plt

nrows, ncols = 50, 65
path = '../img'

#Carga la imagen a una matriz.
def bmp2arr(filename):
    aux = iio.imread(os.path.join(path,filename))
    img = correct_color(aux)
    return correct_orientation(img)

#Corrige el color y rellena espacios.
def correct_color(img):
    aux = np.empty( [nrows,ncols] )
    aux.fill( ~img[0, img.shape[1] - 1] )
    aux[:img.shape[0],:img.shape[1]] = ~img
    return aux

#Corrige el orden de la matriz
def correct_orientation(img):
    return img[::-1]

#Convierte una matriz en escala de grises a un array de una dimension con 1
#si es negro, -1 si es blanco.
def img2learn(img, umbral = 255/2):
    img_flat = img.reshape([nrows*ncols,1])
    return color2binary_v(img_flat, umbral)

#Crea la matriz de Hebb. 
def ley_hebb_gral(filenames):
    size = nrows * ncols
    W = np.zeros( [size, size] )
    
    for im in filenames:
        p = img2learn( bmp2arr(im) )
        Mp = p.dot(p.transpose())
        W += Mp
        
    W -= np.identity( p.shape[0] ) * len(filenames)
    
    return W   

#Función signo.
def sign(value):
    return -1 if value < 0 else 1

#Ecuación de evolución de un estado.
def natural_evolution(HebbMatrix, state_v, index):
    s = HebbMatrix[index].dot( state_v  ) 
    #No necesito restar los valores Wii, porque es nula la diagonal de W
    return sign( s.sum() )

#Pasa de valores {-1,1} a {0,255}
def binary2color(value):
    return 255 if value == 1 else 0

#Pasa de valores [0,255] a {-1,1}
def color2binary(state, umbral):
    return -1 if state < umbral else 1

#Pasa un vector unidimensional con valores de [0,255] a {-1. 1}
def color2binary_v(state_v, umbral = 255/2):
    for index in range(0,state_v.shape[0]):
        state_v[index] = color2binary( state_v[index], umbral)
    return state_v

#Crea una imagen con ruido, donde intensity es un valor entre 0 y 1
#y que permite regular el ruido impuesto.
def img_with_noise(filename, intensity):
    img = bmp2arr(filename)
    noise = np.random.rand(img.shape[0], img.shape[1])
    img *= noise
    img += noise * 255 * intensity     
    return img.reshape( [nrows*ncols, 1] )

# Reconstruye la imagen por medio de la evolución natural.
def reconstruct_img(HebbMatrix,img):
    indexes = np.arange(0, img.shape[0] -1)
    np.random.shuffle(indexes)
    for index in indexes:
        img[index] = natural_evolution(HebbMatrix, img, index)

# Reconstruye parcialmente la imagen, ya que interrumpe le proceso una vez
#pasado por la mitad de los indices
def partial_reconstruct_img(HebbMatrix,img):
    indexes = np.arange(0, img.shape[0] -1)
    np.random.shuffle(indexes)
    tope = indexes[ int(img.shape[0]/2) ]
    for index in indexes:
        img[index] = natural_evolution(HebbMatrix, img, index)
        if tope == index: break

#Funcion para borrar parcialmente una imagen.   
def partial_image(image, orientation, percentage):
    img = img2learn( bmp2arr(image) ).reshape([nrows, ncols])
    if(orientation == 'H'):
        img = partial_image_horizontal(img, percentage)      
    elif(orientation == 'V'):
        img = partial_image_vertical(img, percentage)
    elif(orientation == 'HV'):
        img = partial_image_horizontal(img, percentage)      
        img = partial_image_vertical(img, percentage)
    
    return img.reshape( [nrows * ncols, 1] ) 

#Funcion para borrar una porción vertical de una imagen
def partial_image_vertical(image, percentage):
    image[:image.shape[0], int(image.shape[1] * (1 - percentage/100 ))+1:].fill(-1)
    return image

#Funcion para borrar una porción horizontal de una imagen
def partial_image_horizontal(image, percentage):
    image[int(image.shape[0] * (1 - percentage/100) + 1 ):,:image.shape[1]].fill(-1)
    return image 

#Funcion para leer los nombres de los archivos .bmp desde un archivo de texto.
def read_images(filename):
    #Leo los nombres de las imagenes
    path_images = open(os.path.join(path,filename), "r")
    return path_images.read().split('\n')[0:-1]


#Ploteo dado un vector unidimensional
def plot_vector(vector, axes, row, col):
    vector = vector.reshape([nrows, ncols])
    axes[row,col].imshow( vector, cmap='binary', interpolation='nearest', origin='lower')
    axes[row,col].axis('off')

#Dado un vector de nombres de imagenes, devuelve una matriz donde cada fila 
#es el vector de estados de la imagen
def images_matrix(images):
    img = np.ones([nrows*ncols,1])
    for image in images:
        img = np.append( img, img2learn( bmp2arr(image) ), axis = 1)
    img = np.delete( img, 0, axis = 1)    
    return img.transpose()

#Crea un grafico con subplots, y grafica 3 instantes de la convergencia de la imagen.
#Obs: Una fila del images_array debe ser el vector de estados de la imagne
def plot_evolution( images_array, HebbMatrix ):
    fig, axs = plt.subplots( images_array.shape[0], 3)
    axs = axs.reshape( [images_array.shape[0],3] )
    indices = np.arange(0, images_array.shape[0])
    axs[0,0].set_title('Estado inicial')
    axs[0,1].set_title('Estado intermedio')
    axs[0,2].set_title('Estado final')
    
    for image, row in zip(images_array, indices):
        #Ploteo la imagen inicial      
        plot_vector(image, axs, row, 0)
       
        #Reconstruir parcialmente la imagen 
        partial_reconstruct_img(HebbMatrix, image)
        #Ploteo      
        plot_vector(image, axs, row, 1)
        
        
        #Reconstruir la imagen 
        reconstruct_img(HebbMatrix, image)  
        #Ploteo      
        plot_vector(image, axs, row, 2)

#A partir de una imagen, crea una matriz con alteraciones de esa imagen.
def matrix_alteraciones_img(image):
    img = np.empty([nrows, ncols])
    
    img = img_with_noise(image, 0.35)
    img = color2binary_v(img, 255/5)
    
    img = np.append( img, partial_image(image, 'H', 50), axis = 1)
    img = np.append( img, partial_image(image, 'V', 50), axis = 1)
    img = np.append( img, partial_image(image, 'HV', 50), axis = 1)
   
    return img.transpose()