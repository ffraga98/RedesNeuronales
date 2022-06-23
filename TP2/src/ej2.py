import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions_ej1 import entrenar_perceptron_simple

""" EJERCICIO 2
    Determinar númericamente cómo varia la capacidad del perceptrón simpole en
función del número de patrones enseñados

UN problema es un conjunto de funciones f(input) = output generados aleatoriamente.
    El input es un vector de dimensión N, que son las entradas del perceptron
    El output es un valor escalar.

La dimensión de los vectores no debería afectar a la capacidad.

Se supone que a medida que yo aumente la cantidad de funciones que tiene 
un problema, la capacidad disminuye.

En el caso de la AND, un problema es una fila de la tabla de verdad cuya salida
es el resultado de la operacion. x = 1 1 0, y = 0.

"""

N = 8 #Dimension de las entradas
nro_problemas = 300
nro_patrones = 40
capacidades = np.empty([0,0])

#p = cantidad de patrones.
for p in range(1,nro_patrones+1):
    i = 0
    aprendidos = 0
    print("Cantidad Patrones:", p)
    # Itero 10 veces, es decir, va a haber 10 problemas con la misma cantidad de patrones.
    while(i < nro_problemas):
        print("Problema", i, " de ", nro_problemas, " con ", p, " patrones." )
        #Este es UN problema.
        entradas = np.random.randint(0, high = 2, size = [p, N])
        salida = np.random.randint(0, high = 2, size = [p,1])
        
        w_final, aprendido = entrenar_perceptron_simple(entradas, salida)
        if( aprendido ):
            aprendidos += 1
        
        i += 1
        
    capacidad = aprendidos/nro_problemas
    print('Capacidad', capacidad)
    capacidades = np.append(capacidades, capacidad)
    
fig = plt.figure()
plt.plot( np.arange(1, nro_patrones + 1) , capacidades  )
plt.title(r"Capacidad en funcion de la cantidad de patrones con %d problemas." %nro_problemas)
plt.ylabel("Capacidad")
plt.xlabel("Numero de patrones")
plt.grid(color = "0.8", linestyle = "--")

