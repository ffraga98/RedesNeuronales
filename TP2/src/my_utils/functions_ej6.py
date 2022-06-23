import numpy as np
from my_utils.functions import signo, inicializar_W
from my_utils.functions_ej3 import calcular_salida_real,error_salida, calcular_salidas, backdrop_propagation, calcular_salida_real, error
from math import exp

TEMPERATURA_INICIAL = 20
ALFA = 0.95
BETA = 1
ERROR_MAX = 0.01
ITERACIONES_MAX = 5e3
ETA = 0.05

def sortear_actualizacion(dE, T):
    return np.random.binomial(n = 1, p = probabilidad(dE,T))
    
def probabilidad(dE,T):
    res = exp(-dE/(BETA*T))
   
    return res

def entrenar_perceptron_multicapa_SA( entradas, salidas_deseadas, capas_perceptron ):
    # Paso a valores 1 y -1.
    signo(entradas)
    signo(salidas_deseadas)
    
    #Inicializo aleatoriamente la matriz de pesos sinapticos de todas las capas.
    W = inicializar_W(capas_perceptron)

    iteracion = 0
    error_ = 1
    errores = np.array([])
    print("Entrenando...")
    
    T = TEMPERATURA_INICIAL
    salidas_r = calcular_salida_real(W, entradas)
    error1 = error(salidas_deseadas, salidas_r)
    # Mientras no se supere las iteraciones o el error no sea menor al ERROR_MAX...
    while (error_ > ERROR_MAX and iteracion < ITERACIONES_MAX) : 
        sum_deltas = np.zeros(W.shape, dtype = object)
        salidas_r = np.array([])
        entradas_salidas = list(zip(entradas, salidas_deseadas))
        #Para cada par patron-salida
        
        dW_aux = np.zeros( W.shape, dtype = object )
        dW_aux[0] = np.array( np.random.normal(0,0.5, W[0].shape), dtype = object)      
        dW_aux[1] = np.array( np.random.normal(0,0.5, W[1].shape), dtype = object)
        W_aux = W + dW_aux
        
        salidas_r = calcular_salida_real(W, entradas)
        salidas_r_aux = calcular_salida_real(W_aux, entradas)
        
        error_ =  error(salidas_deseadas, salidas_r)
        error_aux =  error(salidas_deseadas, salidas_r_aux)
        delta_error = error_aux - error_

        if( delta_error < 0 or sortear_actualizacion(delta_error, T)):
            #Actualizamos la W
            W = W_aux
            error_ = error_aux

        iteracion += 1
        T = ALFA * T
        print(f"Temperatura: {T}")
        errores = np.append( errores, error_)

        print(f"Error: {error_}")
    return W, errores

