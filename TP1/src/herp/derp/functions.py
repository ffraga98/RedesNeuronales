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

# Numero de columnas y filas que tienen todas las imagenes.
nrows, ncols = 50, 65
path = '../img'


# Crea la matriz de Hebb a partir de una matriz donde las filas son las imagenes
#a aprender. 
def ley_hebb_gral(images):
    # Las imagenes deben estar en FILA.
    size = images.shape[1]
    W = np.zeros( [size, size] )
    
    for p in images:
        p = p.reshape([1,size])
        Mp = p.transpose().dot( p )
        W += Mp
        W -= np.identity( p.shape[1] )
    
    return W   

#Ecuación de evolución de un estado.
def natural_evolution(HebbMatrix, state_v, index):
    state_v = np.reshape(state_v, [state_v.shape[0],1])
    s = HebbMatrix[index].dot( state_v  ) 
    #No necesito restar los valores Wii, porque es nula la diagonal de W
    return color2binary( s.sum(), 0 )

# Reconstruye la imagen por medio de la evolución natural de manera asincronica.
# Trabaja por referencia, por lo que no devuelve un vector nuevo.
def reconstruct_asyn(HebbMatrix,img, proportion_of_index = 1):
    indexes = np.arange( img.shape[0] )
    np.random.shuffle(indexes)
    indexes = indexes[:int(proportion_of_index*indexes.shape[0])]
    for index in indexes:
        img[index] = natural_evolution(HebbMatrix, img, index)

# Crea una matriz de 1 y -1 alternados. 
# Creada para pisar romper informacion almacenada en imagenes.
def uniform_noise_matrix(nrows, ncols):
    return np.fromfunction(lambda i, j: 2*((i + j)%2)-1 , (nrows, ncols), dtype=int)

#Convierte una matriz en escala de grises a un array donde pone 1
#si los valores son mayores al umbral, -1 en caso contrario. 
def img2learn(img, umbral = 255/2):
    img_flat = img.flatten()
    for index in range(img_flat.shape[0]):
        img_flat[index] = color2binary( img_flat[index], umbral)
        
    return np.reshape(img_flat, [img.shape[0], img.shape[1]]) #Lo devuelve en su tamaño original..

#Convierte los valores de la matriz menores a 0 en -1 y los mayores a 1.
def signo_vectorial(img):
    return img2learn(img, 0)

#Convierte un valor entre [0,255] a {-1,1}, segun el umbral.
#Si le pasamos umbral = 0, se convierte en la funcion signo.
def color2binary(state, umbral):
    return -1 if state < umbral else 1

#Crea una imagen con ruido, donde intensity es un valor entre 0 y 1
#y que permite regular el ruido impuesto.
def img_with_noise(filename, intensity):
    img = bmp2arr(filename)
    noise = np.random.rand(img.shape[0], img.shape[1])
    img *= noise
    img += noise * 255 * intensity     
    return img.reshape( [1, nrows*ncols] )


#Dado un vector de nombres de imagenes, devuelve una matriz donde cada fila 
#es el vector de estados de la imagen
def images_matrix(images):
    img = np.empty([1,nrows*ncols])
    for image in images:
        img = np.append( img, img2learn( bmp2arr(image) ), axis = 0)
    # img = np.delete( img, 0, axis = 0)    
    return img

#A partir de una imagen, crea una matriz con alteraciones de esa imagen.
def matrix_alteraciones_img(image):
    img = np.empty([nrows, ncols])
    
    img = img_with_noise(image, 0.35)
    img = img2learn(img, 255/5)
    
    img = np.append( img, partial_image(image, "H", 50), axis = 0)
    img = np.append( img, partial_image(image, "V", 50), axis = 0)
    img = np.append( img, partial_image(image, "HV", 50), axis = 0)
   
    return img

#Funcion para borrar una porción una imagen
def partial_image(image, orientation, percentage):
    proportion = percentage/100
    image = img2learn( bmp2arr(image) ).reshape([nrows, ncols])
    noise = uniform_noise_matrix(nrows, ncols)
    if 'V' in orientation:
        image[:image.shape[0], int(image.shape[1] * (1 - proportion ))+1:] = noise[:image.shape[0], int(image.shape[1] * (1 - proportion ))+1:]
    
    if 'H' in orientation:
        image[int(image.shape[0] * (1 - proportion) + 1 ):, :image.shape[1]] = noise[int(image.shape[0] * (1 - proportion) + 1 ):,:image.shape[1]]
    
    return image.reshape([1,nrows*ncols])

#Crea un grafico con subplots, y grafica 3 instantes de la convergencia de la imagen.
#Obs: Una fila del images_array debe ser el vector de estados de la imagne
def plot_evolution( images_array, HebbMatrix ):
    fig, axs = plt.subplots( images_array.shape[0], 3)
    axs = axs.reshape( [images_array.shape[0],3] )
    axs[0,0].set_title('Estado inicial')
    axs[0,1].set_title('Estado intermedio')
    axs[0,2].set_title('Estado final')
    
    for row, image in enumerate(images_array):
        #Ploteo la imagen inicial      
        plot_vector(image, axs, row, 0)
       
        #Reconstruir parcialmente la imagen 
        reconstruct_asyn(HebbMatrix, image, 0.5)
        #Ploteo      
        plot_vector(image, axs, row, 1)
        
        #Reconstruir la imagen 
        reconstruct_asyn(HebbMatrix, image)  
        #Ploteo      
        plot_vector(image, axs, row, 2)

#Ploteo dado un vector unidimensional
def plot_vector(vector, axes, row, col, size = [nrows, ncols]):
    vector = vector.reshape(size)
    axes[row,col].imshow( vector, cmap='binary', interpolation='nearest', origin='lower')
    axes[row,col].axis('off')


#Carga la imagen a una matriz.
def bmp2arr(filename):
    aux = iio.imread(os.path.join(path,filename))
    img = correct_color(aux)
    return correct_orientation(img)

#Corrige el color y rellena espacios, debido a la normalización del tamaño de las
#imagenes.
def correct_color(img):
    aux = np.empty( [nrows,ncols] )
    aux.fill( ~img[0, img.shape[1] - 1] )
    aux[:img.shape[0],:img.shape[1]] = ~img
    return aux

#Corrige el orden de la matriz
def correct_orientation(img):
    return img[::-1]

# Crea la matriz de Hebb a partir de una matriz donde filenames es un archivo
#con los nombres de los archivos que contienen las imagenes bmp.
def ley_hebb_from_filenames(filenames):  
    images = np.empty([0,nrows*ncols])
    
    for file in filenames:
        images = np.append(images, img2learn(bmp2arr(file)), axis = 0)
    
    
    W = ley_hebb_gral(images)
    
    return W   

#Funcion para leer los nombres de los archivos .bmp desde un archivo de texto.
def read_images(filename):
    #Leo los nombres de las imagenes
    path_images = open(os.path.join(path,filename), "r")
    return path_images.read().split('\n')[0:-1]
#%% FUNCIONES AGREGADAS PARA EL EJERCICIO 2 y 3

#Retorona el patron reconstruido sincronicamente.
def reconstruct_syn(HebbMatrix, img, proportion_of_index = 1):
    new_state = np.empty(img.shape[0])
    indexes = np.arange( img.shape[0] )
    np.random.shuffle(indexes)
    indexes = indexes[:int(proportion_of_index*indexes.shape[0])]
    for index in indexes:
        new_state[index] = natural_evolution(HebbMatrix, img, index)
        
    return new_state

#Crea un grafico con subplots, y grafica 3 instantes de la convergencia de la imagen.
#Obs: Una fila del images_array debe ser el vector de estados de la imagne
def plot_evolution_syn( images_array, HebbMatrix, size = [nrows, ncols] ):
    fig, axs = plt.subplots( images_array.shape[0], 2)
    axs = axs.reshape( [images_array.shape[0],2] )
    axs[0,0].set_title('Estado inicial')
    axs[0,1].set_title('Estado intermedio')
    # axs[0,2].set_title('Estado final')
    
    for row, image in enumerate(images_array):
        #Ploteo la imagen inicial      
        plot_vector(image, axs, row, 0, size)
       
        #Reconstruir la imagen 
        image = reconstruct_syn(HebbMatrix, image)  
        #Ploteo      
        plot_vector(image, axs, row, 1, size)

#Calcular la cantidad de bits diferentes entre dos patrones.
def estados_erroneos(img_original, img_final):
     return sum( abs(img_original - img_final)/2 )

#Calcula las diferencias de bits promedio en un patron.
def calcular_error_patron(img_original, img_final):
    return estados_erroneos(img_original, img_final)/img_original.shape[0]

#Hace una aproximación lineal con los vectores de igual tamaño que le pases.
# Utilizado para calcular la capacidad.
def calculo_capacidad( n, p ):
   coef_pol = np.polyfit( n, p, 1 )
   return coef_pol[0], coef_pol[1]

#Genera num_patrones vectores aletoriamente de un largo de neuronas.
# Se puede especificar la covarianza, en tal caso se generan normales 
#multivariadas con las covarianzas cruzadas iguales a covarianza.
def generate_random_patterns(num_patrones, neuronas, covarianza = 'None'):
    if(covarianza == 'None'):
        patterns = np.random.rand(num_patrones, neuronas)
        patterns = 2*patterns -1 
    else:
        cov_matriz = (np.ones([neuronas, neuronas]) - np.eye(neuronas)) * covarianza
        cov_matriz += np.eye(neuronas)
        #Cada fila es un patron. 
        patterns = np.random.multivariate_normal([0] * neuronas, cov_matriz, size = num_patrones)
    
    patterns = signo_vectorial(patterns)
    return patterns

# Calcula el numero de neuronas que se necesitaron para que la probabilidad de 
# error estimada sea menor que la pError buscada
def calculo_neuronas_patrones(pError, capacidad_esperada, num_img, covarianza = 'None', porcentaje_olvido = 0):    
    neuronas = np.empty(0)
    for imgs in num_img:
        #Tengo que aumentar las neuronas hasta que el error me de menor a la pError.
        n = calcular_neuronas(pError, capacidad_esperada, imgs , covarianza, porcentaje_olvido)
        neuronas = np.append(neuronas, n)
    
    return neuronas, num_img

#Algoritmo de fuerza bruta para obtener las neuronas que obtengan un error 
# menor a la probabilidad de error.
def calcular_neuronas(pError, capacidad_esperada, num_patrones, covarianza = 'None', porcentaje_olvido = 0):
    neuronas = np.empty(0) 
    
    #Utilizo la capacidad esperada para inicializar la cantidad de neuronas
    # en un numero cercano al esperado.
    n_init = (num_patrones/capacidad_esperada)
    n_init -= n_init/4
    n = int(n_init)

    #Loop para calcular muchas veces la cantidad de neuronas y poder promediar 
    # los resultados.
    for i in range(20):
        error = 1
        while( pError < error):
            n += 1
            for j in range(5): 
                #Intenta 5 veces con la misma cantidad de neuronas.
                error = calcular_error(num_patrones, n, covarianza, porcentaje_olvido)
                if(pError < error): break
            
            if(neuronas.shape[0] != 0): 
                #En caso de que hay una dif mayor a 600, reseteo el contador.
                #Esto se utilizó para evitar que quede eternamente aumentando neuronas, 
                # especialmente en numeros extremadamente altos.
                if( n - neuronas[-1] > 600):
                    n = int(n_init)
        neuronas = np.append(neuronas, n)
        n = int(n_init)

    #Retorno el promedio del numero de las neuronas.
    return np.mean(neuronas)

#Calculo del error para un set de vectores generados pseudoaleatoriamente.
def calcular_error(num_patrones, num_neuronas, covarianza, porcentaje_olvido = 0):
        #Crear matrices pseudo-aleatorias
        noise = generate_random_patterns(num_patrones, num_neuronas, covarianza)

        #Enseñar
        W = ley_hebb_gral(noise)
        #Olvidar si es especificado. (Utilizado en el ejercicio 3)
        if(porcentaje_olvido != 0):
            W = olvidar_aprendizaje(W, porcentaje_olvido)
        
        error = np.empty(0)
        #Calculo del error.
        for p in noise: 
            error = np.append( error, calcular_error_patron( p, reconstruct_syn(W, p) ) )
        
        return np.mean(error)

# Setea en 0 un percentage de elementos wij de la matriz de sinapsis.
def olvidar_aprendizaje(HebbMatrix, percentage):
    newHebbMatrix = HebbMatrix.flatten() #Si no me equivoco devuelve una copia, no la refencia.
    indexes = np.arange( newHebbMatrix.shape[0] )
    np.random.shuffle(indexes)
    indexes = indexes[:int(percentage/100*indexes.shape[0])]
    for i in indexes:
        newHebbMatrix[i] = 0
    return np.reshape(newHebbMatrix, [HebbMatrix.shape[0], HebbMatrix.shape[1]])