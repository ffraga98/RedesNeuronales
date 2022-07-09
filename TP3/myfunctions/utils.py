import numpy as np

MAX_ITERACIONES = 30
ALFA = 0.9
SIGMA = 5
ETA = 0.5
DIMENSION = 2

NEURONAS_ANCHO = 25
NEURONAS_ALTO = 25

def indice2D(indice):
       fil = int(indice / NEURONAS_ANCHO)
       col = int(indice - NEURONAS_ANCHO * fil)
       return fil,col
   
#Distancia entre las neuronas del mapa de resultados
def inicializar_W(ancho,alto):
    return ( 0.4 * np.random.rand(ancho,alto,DIMENSION) - 0.2 )

def norma(x):
    return np.sqrt(sum( x**2 ))  

def cuadrado_unitario(cantidad):
    patrones = np.zeros([cantidad,DIMENSION])
    for i in range(cantidad):
        patron = (2 * np.random.rand(1,DIMENSION) - 1 ) 
        patrones[i,:] = patron
        
    return patrones

def circulo_unitario(cantidad):
    patrones = np.zeros([cantidad,DIMENSION])
    
    i = 0
    while( i < cantidad):
        patron = (2 * np.random.rand(1,DIMENSION) - 1 ) 
        if( norma( patron[0] )  <=  1):
            patrones[i,:] = patron
            i += 1
            
    return patrones

def distancia_entre_neuronas(W):
    ancho = W.shape[0]
    alto = W.shape[1]
    distancia_neuronas = np.zeros((ancho, alto), dtype = object)
    for i in range(ancho):
        for j in range(alto):
            distancia = (np.array([np.arange(ancho)]).transpose() - i )**2 + ( np.array([np.arange(alto)]) - j )**2
            distancia_neuronas[i,j] = np.sqrt(distancia)
            
    return distancia_neuronas

def obtener_distancia_pesos(W, patron):
    distancia_pesos = np.array([])
    
    for i in W:
        for j in i:
            distancia = norma(j - patron)
            distancia_pesos = np.append(distancia_pesos, distancia);
    
    return distancia_pesos

def calcular_vecindad(sigma, distancia, dim):
    vecindad = np.exp( -( distancia ** 2)/(2 * sigma **2) )
    
    return vecindad

def calcular_dW(W,patron,vecindad):
    dW = np.zeros([NEURONAS_ANCHO,NEURONAS_ALTO,DIMENSION])
    for i in range(DIMENSION):
        dW[:,:,i] = ETA * vecindad * ( patron[i] - W[:,:,i] )
        
    return dW

def neurona_ganadora(distancia):
    index_ganadora = np.where( distancia ==  min(distancia) );
    #En caso de haber mas de un ganador elegimos aleatoriamente
    i = int( len( index_ganadora[0] ) * np.random.rand() )
    return indice2D( index_ganadora[0][i] )

def kohonen(patrones, W):
    distancia_neuronas = distancia_entre_neuronas(W)
    sigma = SIGMA
    W_inicial = np.copy(W)
    print("Aprendizaje")
    for iteracion in range(MAX_ITERACIONES):
        np.random.shuffle(patrones)
    
        for patron in patrones:
            distancia = obtener_distancia_pesos(W, patron)
            
            fila,columna = neurona_ganadora(distancia)
            
            d_neuronas = distancia_neuronas[fila , columna]
        
            vecindad = calcular_vecindad(sigma, d_neuronas, DIMENSION)

            dW = calcular_dW(W, patron, vecindad)
            
            W += dW
        
        if iteracion == int(MAX_ITERACIONES/3):
            W_intermedia = np.copy(W);
            
        sigma *= ALFA
        print(iteracion)    
        
    return np.array([W_inicial, W_intermedia, W])


    
