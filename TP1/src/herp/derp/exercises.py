from functions import read_images, color2binary_v, img2learn, bmp2arr, ley_hebb_gral, reconstruct_img, img_with_noise, partial_image, partial_reconstruct_img
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
        
        #Ploteo la imagen inicial
        init_img = init_img.reshape([ nrows, ncols])
        axs[0].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower')
        axs[0].set_title('Estado inicial')
        
        #Reconstruir parcialmente la imagen 
        init_img = init_img.reshape( [nrows*ncols, 1] )
        partial_reconstruct_img(W, init_img)
        #Ploteo
        init_img = init_img.reshape([nrows, ncols])
        axs[1].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower')
        axs[1].set_title('Estado intermedio')
        
        
        #Reconstruir la imagen 
        init_img = init_img.reshape( [nrows*ncols, 1] )
        reconstruct_img(W, init_img)  
        #Ploteo      
        init_img = init_img.reshape([nrows, ncols])
        axs[2].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower')
        axs[2].set_title('Estado final')
        
        
        
def evolucion_version_alterada(filename):
    images =  read_images(filename)
    
    #Crear una matriz W
    W = ley_hebb_gral(images)
            
    #Pasarle 4 versiones de un imagen (Ruido, V, H, HV)
    #Graficar estado inicial, intermedio y final de cada version de imagen
    
    for img in images:
        fig, axes = plt.subplots(4,3)
        resultados_version_alterada(W, img, 'ruido',axes)
        resultados_version_alterada(W, img, 'H', axes)
        resultados_version_alterada(W, img, 'V', axes)
        resultados_version_alterada(W, img, 'HV', axes)        


def resultados_version_alterada(HebbMatrix, image, alteracion, axs):
    init_img = np.empty([nrows, ncols])
    
    if(alteracion == 'ruido'):
        init_img = img_with_noise(image, 0.5)
        init_img = color2binary_v(init_img, 255/5)    
        axs[0,0].set_title('Estado inicial')
        axs[0,1].set_title('Estado intermedio')
        axs[0,2].set_title('Estado final')
        row = 0
    elif(alteracion == 'H'):
        init_img = partial_image(image, 'H', 50)
        row = 1
    elif(alteracion == 'V'):
        init_img = partial_image(image, 'V', 50)
        row = 2
    elif(alteracion == 'HV'):
        init_img = partial_image(image, 'HV', 50)
        row = 3
        
    #Ploteo la imagen inicial
    init_img = init_img.reshape([ nrows, ncols])
    axs[row,0].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower') 
    axs[row,0].axis('off')
          
        
    #Reconstruir parcialmente la imagen 
    init_img = init_img.reshape( [nrows*ncols, 1] )
    partial_reconstruct_img(HebbMatrix, init_img)
    #Ploteo
    init_img = init_img.reshape([nrows, ncols])
    axs[row,1].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower')
    axs[row,1].axis('off')
    
    
    #Reconstruir la imagen 
    init_img = init_img.reshape( [nrows*ncols, 1] )
    reconstruct_img(HebbMatrix, init_img)  
    #Ploteo      
    init_img = init_img.reshape([nrows, ncols])
    axs[row,2].imshow( init_img, cmap='binary', interpolation='nearest', origin='lower')
    axs[row,2].axis('off')
    
    
    
# def estados_espureos():
#     # No se todavia bien que pretenden de este ejericio
#     # Leer: Spurious States, en la sección 2.2, Hertz, Krogh & Palmer, pág. 24 

def minimo_local_desde_original(filename):
    images =  read_images(filename)
    
    #Crear una matriz W
    W = ley_hebb_gral(images)
    
    #Pasarle 4 versiones de un imagen (Ruido, V, H, HV)
    #Graficar estado inicial, intermedio y final de cada version de imagen
    fig, axs = plt.subplots( len(images), 3)
    indices = np.arange(0, len(images))
    axs[0,0].set_title('Estado inicial')
    axs[0,1].set_title('Estado intermedio')
    axs[0,2].set_title('Estado final')
    for image, i in zip(images, indices):
        img = img2learn( bmp2arr(image) )
       
        #Ploteo la imagen inicial
        img = img.reshape( [nrows, ncols] )
        axs[i,0].imshow( img, cmap='binary', interpolation='nearest', origin='lower') 
        axs[i,0].axis('off')
       
       
        #Reconstruir parcialmente la imagen 
        img = img.reshape( [nrows*ncols, 1] )
        partial_reconstruct_img(W, img)
        #Ploteo
        img = img.reshape([nrows, ncols])
        axs[i,1].imshow( img, cmap='binary', interpolation='nearest', origin='lower')
        axs[i,1].axis('off')
        
        
        #Reconstruir la imagen 
        img = img.reshape( [nrows*ncols, 1] )
        reconstruct_img(W, img)  
        #Ploteo      
        img = img.reshape([nrows, ncols])
        axs[i,2].imshow( img, cmap='binary', interpolation='nearest', origin='lower')
        axs[i,2].axis('off')
        
