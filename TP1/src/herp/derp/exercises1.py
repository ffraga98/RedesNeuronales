from herp.derp.functions import img2learn, signo_vectorial, images_matrix, plot_evolution, matrix_alteraciones_img, nrows, ncols, read_images, ley_hebb_from_filenames
import numpy as np



#EJERCICIO A.
# Aprende una imagen a la vez e intenta reconstruirla desde el ruido.
# Obs: Muchas veces en la reconstrucción converge al inverso, 
# el cual es estado espureo
def verificar_aprendizaje(filename):
    #Lectura de los nombres de archivos
    images =  read_images(filename)
    
    #Crear una matriz W para cada imagen.
    for img in images:
        img = np.array([img])
        W = ley_hebb_from_filenames(img)
        
        #Crear una matriz ruidosa
        init_img = 2*np.random.rand(1, nrows * ncols)-1
        
        plot_evolution(init_img, W)


#EJERCICIO B.
# Intenta reconstruir las imagenes aprendidas partiendo de distintas alteraciones
# de la imagen original.
def evolucion_version_alterada(filename):
    images =  read_images(filename)
    
    #Crear una matriz W
    W = ley_hebb_from_filenames(images)
    patterns = np.empty([0,W.shape[0]])
    for img in images:
        patterns = np.append(patterns, matrix_alteraciones_img(img), axis = 0)
        plot_evolution( patterns, W )
        patterns = np.empty([0,W.shape[0]])


# EJERCICIO C.
#Aprende las imagenes que le pase y grafica la evolucion partiendo desde la
# combinación lineal de las imagenes que le pases.
def verificar_estados_espureos(filename):
    images = read_images(filename)
    W = ley_hebb_from_filenames(images)
    
    images = images_matrix(images)
    #Creamos combinaciones lineales de las imagenes aprendidas.
    img = (  - images[0] + images[1] + images[2] )
    img = signo_vectorial( img.reshape([1, nrows*ncols]) )
    images = np.append(images, img, axis = 0)
    
    img = (  images[0] - images[1] - images[2] )
    img = signo_vectorial( img.reshape([1, nrows*ncols]) )
    images = np.append(images, img, axis = 0)
    
    img = ( - images[0] - images[1] + images[2] )
    img = signo_vectorial( img.reshape([1, nrows*ncols]) )
    images = np.append(images, img, axis = 0)
    
    img = (  images[0] + images[1] - images[2] )
    img = signo_vectorial( img.reshape([1, nrows*ncols]) )
    images = np.append(images, img, axis = 0)
     
    # Agregamos las opuestas.       
    images[0] = ( images[0] * -1 ).reshape([1, nrows*ncols])
    images[1] = ( images[1] * -1 ).reshape([1, nrows*ncols])
    images[2] = ( images[2] * -1 ).reshape([1, nrows*ncols])
    
    plot_evolution(images, W)
    
    
#EJERCICIO D.
# Aprende todas las imagenes que le pases y evalua su evolución
#desde la imagen original.

def minimo_local_desde_original(filename):
    images = read_images(filename)
    
    #Crear una matriz W y cargar una matriz con imagenes
    W = ley_hebb_from_filenames(images)
    images = images_matrix(images)
    #Graficar estado inicial, intermedio y final de cada version de imagen
    plot_evolution(images, W)



