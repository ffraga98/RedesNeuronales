import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions_ej4 import entrenar_perceptron_multicapa, plot_errores
#%%
""" EJERCICIO 4

a. Implemente una red con aprendizaje BackPropagation que aprnda la siguiente 
función
                     f(x,y,z) = sin(x) + cos(y) + z
  donde: x e y pertenecen a [0, 2pi] y z pertence a [-1,1]. 
  Para ellos construya un conjunto de datos de entrenamiento y conjunto de 
  evaluación. Muestre el error un función de las épocas de entrenamiento.
  
b. ¿Cómo varía el número de iteraciones necesarias en función del tamaño 
del minibatch, y de la constante de aprendizaje? 

¿Y el tiempo total de entrenamiento?

"""

CAPAS_OCULTAS_PERCEPTRON = np.array([ 12 ])
#%%
PATRONES_POR_BATCH = 200

e_epoca, e_iteracion = entrenar_perceptron_multicapa(CAPAS_OCULTAS_PERCEPTRON, PATRONES_POR_BATCH)
plot_errores(e_epoca,e_iteracion, PATRONES_POR_BATCH)

PATRONES_POR_BATCH = 100

e_epoca, e_iteracion = entrenar_perceptron_multicapa(CAPAS_OCULTAS_PERCEPTRON, PATRONES_POR_BATCH)
plot_errores(e_epoca,e_iteracion, PATRONES_POR_BATCH)
#%%
PATRONES_POR_BATCH = 50

e_epoca, e_iteracion = entrenar_perceptron_multicapa(CAPAS_OCULTAS_PERCEPTRON, PATRONES_POR_BATCH)
plot_errores(e_epoca,e_iteracion, PATRONES_POR_BATCH)



"""
Notas de la práctica.

Gradiente descendiente ESTÓCASTICO. 
    -> A diferencia del ejercicio 3, que actualizamos una vez pasado TODOS
    los patrones, al tener tantas entradas, voy a fraccionar mis set de datos
    en minibatch (lotes).
    
    -> A medida que completo un minibatch, actualizo la red. El minibatch es
    generado aleatoriamente. Una vez que se haya pasado por todos los lotes, 
    se termina una epoca.
    
Paso a paso:
    1. Generar los conjuntos finitos de vectores de entrenamiento y testeo.
        - Deben estar en el cubo [0, 2pi]x[0, 2pi]x[-1,1]  
    2. Elijo un vector del set de entrenamiento al azar. Computo y aplico deltaW
    3. Repetir paso (2) hasta pasar por cada vector de entrenamiento.
    4. Calculo el error promedio E sobre todo el set de testeo
    5. Repito 2-4 hasta alcanzar un Emin.
    6. Graficar:
        Etrain vs iteracion
        Etest vs epoca -> puede que no sean muy parecida la evolucion del error
                        por el hecho de tener Overfitting. (modelo muy simple
                        con un set muy grande de datos).
        Yreal vs Ydeseada
        
E = 0.002.
1 capa oculta 16 neuronas.
5e5 iteraciones
500 epocas

Variables: 
    #Neuronas en la capa oculta
    Distribucion de los pesos iniciales
    Constante de aprendizaje
    #set de entrenamiento

"""