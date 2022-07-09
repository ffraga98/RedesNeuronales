import numpy as np

MAX_ITERACIONES = 40
ALFA = 0.9
SIGMA = 100
ETA = 0.5
DIMENSION = 2

NEURONAS_LARGO = 25*25

  
#Distancia entre las neuronas del mapa de resultados
def inicializar_W(largo):
    return ( 0.5 * np.random.rand(largo,DIMENSION) - 0.25 )

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
    largo = W.shape[0]
    distancia_neuronas = np.zeros((largo), dtype = object)
    for i in range(largo):
            distancia = (np.array([np.arange(largo)]).transpose() - i )**2 
            distancia_neuronas[i] = np.sqrt(distancia)
            
    return distancia_neuronas

def obtener_distancia_pesos(W, patron):
    distancia_pesos = np.array([])
    
    for i in W:
            distancia = norma(i - patron)
            distancia_pesos = np.append(distancia_pesos, distancia);
    
    return distancia_pesos

def calcular_vecindad(sigma, distancia, dim):
    vecindad = np.exp( -( distancia[0].flatten() ** 2)/(2 * sigma **2) )
    
    return vecindad

def calcular_dW(W,patron,vecindad):
    dW = np.zeros([NEURONAS_LARGO,DIMENSION])
    for i in range(DIMENSION):
        dW[:,i] = ETA * vecindad.flatten() * ( patron[i] - W[:,i] )
        
    return dW

def neurona_ganadora(distancia):
    index_ganadora = np.where( distancia ==  min(distancia) );
    #En caso de haber mas de un ganador elegimos aleatoriamente
    i = int( len( index_ganadora[0] ) * np.random.rand() )
    return index_ganadora

def kohonen(patrones, W):
    distancia_neuronas = distancia_entre_neuronas(W)
    sigma = SIGMA
    W_inicial = np.copy(W)
    print("Aprendizaje")
    
    
    
    for iteracion in range(MAX_ITERACIONES):
        np.random.shuffle(patrones)
        
        for patron in patrones:
            distancia = obtener_distancia_pesos(W, patron)
            
            i = neurona_ganadora(distancia)
            
            d_neuronas = distancia_neuronas[i]
        
            vecindad = calcular_vecindad(sigma, d_neuronas, DIMENSION)

            W += calcular_dW(W, patron, vecindad)
            
        
        if iteracion == int(MAX_ITERACIONES/3):
            W_intermedia1 = np.copy(W);
        if iteracion == int(MAX_ITERACIONES/2):
            W_intermedia2 = np.copy(W);
            
        sigma *= ALFA
        print(iteracion)    
        
    return np.array([[W_inicial, W_intermedia1] , [W_intermedia2, W]])

    
