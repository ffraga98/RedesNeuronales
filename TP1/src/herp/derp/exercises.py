from herp.derp.functions import read_images, color2binary_v, img2learn, bmp2arr, ley_hebb_gral, reconstruct_img, img_with_noise, partial_image, partial_reconstruct_img
import numpy as np
import matplotlib.pyplot as plt


nrows, ncols = 50, 65

def verificar_aprendizaje(filename):
    images =  read_images(filename)
    #Crear una matriz W para cada imagen.
    for img in images:
        img = np.array([img])
        W = ley_hebb_gral(img)
        
        #Crear una matriz ruidosa
        init_img = color2binary_v( np.random.rand(nrows * ncols, 1) * 255 )
        
        fig, axs = plt.subplots(1,3)
        axs = axs.reshape([3,1])
        #Ploteo la imagen inicial
        plot_vector(init_img, axs, 0, 0)
        axs[0,0].set_title('Estado inicial')
        
        #Reconstruir parcialmente la imagen 
        partial_reconstruct_img(W, init_img)
        #Ploteo
        plot_vector(init_img, axs, 1, 0)
        axs[1,0].set_title('Estado intermedio')
            
        #Reconstruir la imagen 
        reconstruct_img(W, init_img)  
        #Ploteo      
        plot_vector(init_img, axs, 2, 0)
        axs[2,0].set_title('Estado final')
        

        
def evolucion_version_alterada(filename):
    images =  read_images(filename)
    
    #Crear una matriz W
    W = ley_hebb_gral(images)
            
    #Pasarle 4 versiones de un imagen (Ruido, V, H, HV)
    #Graficar estado inicial, intermedio y final de cada version de imagen
    for img in images:
        fig, axes = plt.subplots(4,3)
        axes[0,0].set_title('Estado inicial')
        axes[0,1].set_title('Estado intermedio')
        axes[0,2].set_title('Estado final')
        plot_version_alterada(W, img, 'ruido',axes)
        plot_version_alterada(W, img, 'H', axes)
        plot_version_alterada(W, img, 'V', axes)
        plot_version_alterada(W, img, 'HV', axes)        


# def verificar_estados_espureos():
    #Hay alguna forma de evaluar los estados espureos individualmente?
    # Podría probar que aprendiendo una imagen tengo dos estados.
    # Al entrenar 3...?
    

def plot_version_alterada(HebbMatrix, image, alteracion, axs):
    img = np.empty([nrows, ncols])
    
    if(alteracion == 'ruido'):
        img = img_with_noise(image, 0.35)
        img = color2binary_v(img, 255/5)    
        row = 0
    elif(alteracion == 'H'):
        img = partial_image(image, 'H', 50)
        row = 1
    elif(alteracion == 'V'):
        img = partial_image(image, 'V', 50)
        row = 2
    elif(alteracion == 'HV'):
        img = partial_image(image, 'HV', 50)
        row = 3
        
    #Ploteo la imagen inicial
    plot_vector(img, axs, row, 0)
          
        
    #Reconstruir parcialmente la imagen 
    partial_reconstruct_img(HebbMatrix, img)
    #Ploteo
    plot_vector(img, axs, row, 1)
    
    
    #Reconstruir la imagen 
    reconstruct_img(HebbMatrix, img)  
    #Ploteo     
    plot_vector(img, axs, row, 2)
    
    
    
# def estados_espureos():
#     # No se todavia bien que pretenden de este ejericio
#     # Leer: Spurious States, en la sección 2.2, Hertz, Krogh & Palmer, pág. 24 

def minimo_local_desde_original(filename):
    images =  read_images(filename)
    
    #Crear una matriz W
    W = ley_hebb_gral(images)
    
    #Pasarle 4 versiones de un imagen (Ruido, V, H, HV)
    #Graficar estado inicial, intermedio y final de cada version de imagen
    plot_evolution(images, W)
        
def plot_vector(vector, axes, row, col):
    vector = vector.reshape([nrows, ncols])
    axes[row,col].imshow( vector, cmap='binary', interpolation='nearest', origin='lower')
    axes[row,col].axis('off')
    
def plot_evolution(images, HebbMatrix):
    fig, axs = plt.subplots( len(images), 3)
    indices = np.arange(0, len(images))
    axs[0,0].set_title('Estado inicial')
    axs[0,1].set_title('Estado intermedio')
    axs[0,2].set_title('Estado final')
    
    for image, row in zip(images, indices):
        img = img2learn( bmp2arr(image) )
       
        #Ploteo la imagen inicial      
        plot_vector(img, axs, row, 0)
       
        #Reconstruir parcialmente la imagen 
        partial_reconstruct_img(HebbMatrix, img)
        #Ploteo      
        plot_vector(img, axs, row, 1)
        
        
        #Reconstruir la imagen 
        reconstruct_img(HebbMatrix, img)  
        #Ploteo      
        plot_vector(img, axs, row, 2)
