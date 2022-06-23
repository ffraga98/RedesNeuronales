import numpy as np
from my_utils.functions import suma_ponderada, signo, inicializar_W

INDICE_SALIDAS = 0
INDICE_SUMAS = 1
INDICE_DELTAS = 2

#Ejercicio 3.
ETA = 0.1
ERROR_MAX = 0.01
ITERACIONES_MAX = 5e3



def entrenar_perceptron_multicapa( entradas, salidas_deseadas, capas_perceptron ):
    # Paso a valores 1 y -1.
    signo(entradas)
    signo(salidas_deseadas)
    
    #Inicializo aleatoriamente la matriz de pesos sinapticos de todas las capas.
    W = inicializar_W(capas_perceptron)

    iteracion = 0
    error_ = 1
    errores = np.array([])
    print("Entrenando...")
    # Mientras no se supere las iteraciones o el error no sea menor al ERROR_MAX...
    while (error_ > ERROR_MAX and iteracion < ITERACIONES_MAX) : 
        sum_deltas = np.zeros(W.shape, dtype = object)
        entradas_salidas = list(zip(entradas, salidas_deseadas))
        
        #Para cada par patron-salida
        for (entrada,salida_deseada) in entradas_salidas:
            #Calcula las salidas de todas las capas (h, V)
            salidas_capas = calcular_salidas(W, entrada)
            
            #Teniendo todas las sumas ponderadas y las salidas de las neuronas.
            #Calculamos los errores (deltas) correspondiente a cada capa y neurona.
            deltas = backdrop_propagation(ETA, salidas_capas, W, salida_deseada, entrada)
            sum_deltas += deltas
            

        
        #Actualizamos la W
        W += sum_deltas
        iteracion += 1
        
        salidas_r = calcular_salida_real(W, entradas)
        error_ =  error(salidas_deseadas, salidas_r)
        errores = np.append( errores, error_)

        # print(f"Error: {error_}")
    return W, errores


def error_al_modificar_deltaW(deltas_w, pesos_sinapticos, entradas, salidas_deseadas,numeros_capa,numeros_entrada, numeros_salida):
    print("Modificando pesos!")           
    errores = np.array([])

    for dw2 in deltas_w[::-1]:
        pesos_sinapticos[numeros_capa[0]][numeros_salida[0], numeros_entrada[0]] += dw2
        # print(f"Peso fila: {dw2}")
        
            #Modifico W.
        for dw1 in deltas_w:
            pesos_sinapticos[numeros_capa[1]][numeros_salida[1], numeros_entrada[1]] += dw1
            # print(f"Peso sinaptico 2: {pesos_sinapticos[0][numeros_capa[1]][numeros_peso[1]]}")
            
            salidas_r = calcular_salida_real(pesos_sinapticos, entradas)
                
            error_ =  error(salidas_deseadas, salidas_r)
            errores = np.append( errores, error_)
            # print(f"Error: {error_}")
            pesos_sinapticos[numeros_capa[1]][numeros_salida[1], numeros_entrada[1]] -= dw1
            
        pesos_sinapticos[numeros_capa[0]][numeros_salida[0], numeros_entrada[0]] -= dw2
    
    return np.reshape( errores, [deltas_w.size, deltas_w.size] )
    
def func_activacion( h ):
    h = np.array(h, dtype = float)
    return np.tanh( h )

def derivada_func_activacion( h ):
    return 1 - np.tanh( h )**2

def nucleo_procesamiento( w, x):
    suma = suma_ponderada(w, x)
    z = func_activacion( suma )
    return z, suma

# Devuelve un vector de matrices con las salidas y las sumas de cada capa.
def calcular_salidas(pesos_sinapticos, patron):
    info_por_capa = np.zeros(pesos_sinapticos.shape[0], dtype = object)
    
    #Agrego el bias
    patron = np.append(patron, 1)
    
    for index,W in enumerate(pesos_sinapticos):
        if(index == pesos_sinapticos.shape[0]):
            break
        
        salidas, sumas = nucleo_procesamiento(W, patron)
        
        info = np.array( [salidas, sumas] )
        info_por_capa[index] = info
        
        patron = salidas
        #Agrego el bias
        patron = np.append(patron, 1)

    return info_por_capa

def calcular_salida_real(pesos_sinapticos, patrones):
    salida = np.array([])
    for patron in patrones:
        salida = np.append( salida, calcular_salidas(pesos_sinapticos, patron)[-1][INDICE_SALIDAS])
    return salida

def backdrop_propagation(eta, salidas_capas, matrices_W, salida_deseada, entrada):
    #Calculo de deltas de cada capa
    salidas_errores_capas = computar_error_pesos( salidas_capas, salida_deseada, matrices_W)
    #Calculo de Delta w, retorna una matrices configuradas como W.
    return computar_deltas(eta, salidas_errores_capas, entrada )

#  Retorna un vector de matrices que poseen la informacion de salidas, 
# sumas_ponderadas y deltas correspondientes al error
def computar_error_pesos( info_capa, salidas_esperadas, pesos_sinapticos):
    #backdrop propagation, damos vuelta el sentido de los vectores.
    info_capa = info_capa[::-1]
    pesos_sinapticos = pesos_sinapticos[::-1]
    w_capa = 0
    
    for i,info in enumerate(info_capa):
        salidas_capa = np.array(info[INDICE_SALIDAS])
        sumas_capa = np.array(info[INDICE_SUMAS])
        
        if(i == 0):
            # Si i==0, es porque estamos calculando a la salida de la red.
            deltas_sig_capa = error_salida(salidas_esperadas, salidas_capa, sumas_capa)
            deltas_sig_capa = np.reshape(deltas_sig_capa, [1, deltas_sig_capa.shape[0],1])
        else:
            deltas_sig_capa = calcular_error_capa(sumas_capa, w_capa, deltas_sig_capa)
        
        #Siguiente matriz W
        w_capa = pesos_sinapticos[i]
        #Guardamos la informacion de los errores.
        info_capa[i] = np.vstack( (info, deltas_sig_capa) )
         
    #Retornamos en el sentido original
    return info_capa[::-1]

def computar_deltas(eta, info_capas, entradas):
    matrices_deltas = np.zeros(info_capas.shape, dtype = object)

    for i, capa in enumerate(info_capas):
        #Agrego el bias.
        entradas = np.append(entradas, 1)
        salidas = capa[INDICE_SALIDAS]
        errores = capa[INDICE_DELTAS]
        
        matriz_deltas = np.zeros((salidas.shape[0], entradas.shape[0]))

        for j,v in enumerate(entradas):  
            deltas = eta * v * errores
            matriz_deltas[:,j] = deltas.flatten()
            
        matrices_deltas[i] = matriz_deltas
        entradas = np.copy(salidas)         
        
    return matrices_deltas

# Retorna el calculo de las deltas de la salidas, de las neuronas de la ultima
#capa.
def error_salida(salidas_esperadas, salidas, sumas):
    return derivada_func_activacion(sumas) * ( salidas_esperadas - salidas)

# Retorna un vector de deltas de la capa.        
def calcular_error_capa(sumas_capa, w_capa, deltas_i):
     wij = w_capa[:,:-1] #No tomo en cuenta los pesos de los bias
     return derivada_func_activacion(sumas_capa) * wij.transpose() @ deltas_i 


def error(salidas_deseadas, salidas):
    salidas = np.reshape(salidas, salidas_deseadas.shape)
    return 0.5 * sum( (salidas_deseadas - salidas)**2 )

#Eleccion de 2 pesos al azar.
def sortear_pesos(W):
    iguales = True
    while iguales:
        numeros_capa = np.array([], dtype = int)
        numeros_entrada = np.array([], dtype = int)
        numeros_salida = np.array([], dtype = int)
        for i in range(2):
            #Capas de pesos 
            numeros_capa = np.append( numeros_capa, np.random.randint( 0 , W.size ))
            #Neurona de la capa
            numeros_entrada = np.append( numeros_entrada, np.random.randint( 0 , W[numeros_capa[i]][0,:].size ))
            if( numeros_capa[i] == 0):
                numeros_salida = np.append( numeros_salida, np.random.randint( 0 , W[ numeros_capa[i]][:,0].size - 1))
            else:
                numeros_salida = np.append( numeros_salida, np.random.randint( 0 , W[ numeros_capa[i]][:,0].size ))
    
        iguales = ( numeros_salida == numeros_entrada ).all()
        return numeros_capa, numeros_entrada, numeros_salida
    
