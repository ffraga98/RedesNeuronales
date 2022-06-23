import numpy as np
    
# Recibe los dos vectores en forma de fila.
def suma_ponderada( w, x ):
    x = np.reshape( x, [ x.shape[0], 1 ] )
    return w.dot(x)


def signo( elemento ):
    elemento = np.reshape( elemento, [ elemento.shape[0] * elemento.shape[1], 1 ])
    for index, x in enumerate(elemento):
        elemento[index] = 1 if ( x > 0 ) else -1

# Nuevos metodos. (Inyectados por el ej3)
def inicializar_W(capas_perceptron):
        W = np.zeros((capas_perceptron.shape[0] - 1 ), dtype = object )
        for i in range(W.shape[0]):    
            W_capa = np.random.uniform( -0.1, 0.1, (capas_perceptron[i+1], capas_perceptron[i]) )
            W_capa = np.hstack((W_capa, np.ones([capas_perceptron[i+1],1] )))
            W[i] = W_capa
        
        return W

def dec2bin( num ):
    return bin(num).replace("0b","")

def capas_perceptron( entradas, salidas_deseadas, neuronas_capas_ocultas):
    #Agrego neuronas de entrada
    n_entradas = np.array([int(entradas.shape[1])])
    capas = np.append(n_entradas, neuronas_capas_ocultas) 
    #Agrego salidas
    n_salidas = np.array([int(salidas_deseadas.shape[1])])
    capas = np.append(capas, n_salidas)
    return capas

def entradas( numero_entradas ):
    cantidad = 2**numero_entradas
    matriz_entradas = np.zeros( (1,numero_entradas) )
    for i in range(1,cantidad):
        n = np.array([])
        n_bin = dec2bin(i)
        
        while( len(n_bin) != numero_entradas):
            n_bin = "0" + n_bin
            
        for a in str(n_bin):
            n = np.append(n , a)

        n = np.reshape(list(map(int, n)), (1, numero_entradas))
        matriz_entradas = np.append(matriz_entradas, n , axis = 0)
    
    return matriz_entradas


def and_input(entradas):
    resultados = np.array([])
    
    for x in entradas:
         resultados = np.append( resultados, and_( x ))
        
    return np.reshape(resultados, [resultados.shape[0],1])
    
def and_( entrada ):
    cantidad_entradas = len(entrada)
    entrada = np.reshape( entrada, [1,entrada.shape[0]])
    return ( int(  entrada.dot( entrada.transpose() )  / cantidad_entradas ) )


def or_input(entradas):
    resultados = np.array([])
    
    for x in entradas:
         resultados = np.append( resultados, or_( x ))
        
    return np.reshape(resultados, [resultados.shape[0],1])
    
def or_( entrada ):
    entrada = np.reshape( entrada, [1,entrada.shape[0]])
    return  1 if (entrada.dot( entrada.transpose()) > 0) else 0

#Cada fila de las entradas representa un patron
def xor(entradas):
    resultados = np.array([])
    
    for x in entradas:
         resultados = np.append( resultados, xor_( x ))
        
    return np.reshape(resultados, [resultados.shape[0],1])
    
def xor_( entrada ):
    entrada = np.reshape( entrada, [1,entrada.shape[0]])
    return int( ( entrada.dot( entrada.transpose() ) )  % 2)