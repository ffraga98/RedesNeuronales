     
import numpy as np
from my_utils.functions import signo, suma_ponderada

def func_activacion( suma ):
    return signo(suma)

def nucleo_procesamiento( w, x):
    suma = suma_ponderada(w, x)
    func_activacion( suma )
    return suma

def error_salida( salida_deseada, salida):
    return salida_deseada - salida

def correccion_pesos_sinapticos( w, entradas, error ):
    entradas = np.reshape( entradas, [ entradas.shape[0], 1 ] )
    factor_aprendizaje = 0.4
    for peso, x in zip(w.transpose(), entradas):
        peso[0] += factor_aprendizaje * error * x[0]

#Refactorizar, hacer funciones más abstractas.
def entrenar_perceptron_simple( entradas, salida_deseada ):
    aprendido = False
    iteracion = 0
    max_iteraciones = 2000
    
    #Agrego el bias a todas las entradas
    entradas = np.append(np.ones((entradas.shape[0],1)), entradas , axis = 1)
    
    #Crea la matriz w aleatoriamente.
    w = np.random.uniform( -0.2, 0.2, (salida_deseada.shape[1], entradas.shape[1]) )

    # Convierte los vectores en 1 y -1
    signo(entradas)
    signo(salida_deseada)
    
    nro_entrada = 0
    
    #Itera hasta pasar por todas las entradas o se supere las iteraciones.
    while( (nro_entrada != entradas.shape[0]) and (iteracion < max_iteraciones) ):
        iteracion += 1
        
        salida_obtenida =  nucleo_procesamiento( w, entradas[ nro_entrada ])
        error = error_salida( salida_deseada[ nro_entrada ], salida_obtenida )

        if( error ):
            correccion_pesos_sinapticos( w, entradas[ nro_entrada ] , error )
            nro_entrada = 0
        else:
            nro_entrada += 1

    #En caso de superar las iteraciones, devolvemos un array vacío.
    if(iteracion < max_iteraciones):    
        print("Encontrada la solución en ", iteracion, " iteraciones")
        print(w)
        aprendido = True
    else:
        
        print("No se encontró ninguna solución.")
        
    print(" ")
    return w, aprendido