#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 05:50:47 2022

@author: fernando
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio as iio

nrows, ncols = 50, 65
path = '/home/fernando/Documentos/Archivos FIUBA/1C2022/REDES_NEURONALES/TP/TP1/img'


#Carga la imagen a una matriz.
def bmp2arr(filename):
    #Booleanos: img = np.array(Img.open(os.path.join(path,filename)))
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
def img2learn(img):
    img_flat = img.reshape([nrows*ncols,1])
    return color2binary_v(img_flat)

def ley_hebb_gral(filenames):
    size = nrows * ncols
    W = np.zeros( [size, size] )
    
    for im in filenames:
        p = img2learn( bmp2arr(im) )
        Mp = p.dot(p.transpose())
        W += Mp
        
    W -= np.identity( p.shape[0] ) * len(filenames)
    
    return W   
    
def sign(value):
    return -1 if value < 0 else 1

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

def color2binary_v(state_v, umbral = 255/2):
    for index in range(0,state_v.shape[0]):
        state_v[index] = color2binary( state_v[index], umbral)
    return state_v

#intensity es un valor entre 0 y 1
def img_with_noise(filename, intensity):
    img = bmp2arr(filename)
    noise = np.random.rand(img.shape[0],img.shape[1])
    img += noise * 255 * intensity 
    img *= noise    
    return img.reshape( [nrows*ncols, 1] )

def reconstruct_img(HebbMatrix,img):
    indexes = np.arange(0, img.shape[0] -1)
    np.random.shuffle(indexes)
    for index in indexes:
        img[index] = natural_evolution(HebbMatrix, img, index)

def partial_reconstruct_img(HebbMatrix,img):
    indexes = np.arange(0, img.shape[0] -1)
    np.random.shuffle(indexes)
    tope = indexes[ int(img.shape[0]/2) ]
    for index in indexes:
        img[index] = natural_evolution(HebbMatrix, img, index)
        if tope == index: break


def plot_bmp(filename):
    img = bmp2arr(filename)
# Ruido
    noise = np.random.rand(img.shape[0],img.shape[1]) * 255
    img += noise
# creating a plot
    pixel_plot = plt.figure()
    plt.title(filename)
# plotting a plot
    pixel_plot.add_axes()  
# customizing plot
    plt.title("pixel_plot")
    pixel_plot = plt.imshow( img, cmap='binary', interpolation='nearest', origin='lower')
# show plot
    plt.show()
   
#Recibe un vector de largo nrows * ncols!!
def plot_img(img):
    img = img.reshape( [nrows,ncols] )   
    pixel_plot = plt.figure()
    pixel_plot.add_axes()  
    plt.title("pixel_plot")
    pixel_plot = plt.imshow(img, cmap='binary', interpolation='nearest', origin='lower')
    plt.show()
    
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

def partial_image_vertical(image, percentage):
    image[:image.shape[0], int(image.shape[1] * (1 - percentage/100 ))+1:].fill(-1)
    return image

def partial_image_horizontal(image, percentage):
    image[int(image.shape[0] * (1 - percentage/100) + 1 ):,:image.shape[1]].fill(-1)
    return image 


def read_images(filename):
    path = '/home/fernando/Documentos/Archivos FIUBA/1C2022/REDES_NEURONALES/TP/TP1/img'
    #Leo los nombres de las imagenes
    path_images = open(os.path.join(path,filename), "r")
    return path_images.read().split('\n')[0:-1]