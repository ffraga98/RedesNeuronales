import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions import entradas, xor, capas_perceptron
from my_utils.functions_ej6 import entrenar_perceptron_multicapa_SA

''' EJERCICIO 6
Encontrar un perceptrón multicapa que resuelva una XOR de 2 entradas
mediante simulated annealing.
Graficar el error a lo largo del proceso de aprendizaje.
'''

#Numero de entradas 
ENTRADAS = 2
#Capas ocultas
CAPAS_OCULTAS_PERCEPTRON = np.array([7])

x_input = entradas(ENTRADAS)
xor_output = xor(x_input)

#Capas de neuronas del perceptron
capas_perceptron = capas_perceptron(x_input, xor_output, CAPAS_OCULTAS_PERCEPTRON)


#Configuracion del ejercicio
W, errores = entrenar_perceptron_multicapa_SA(x_input, xor_output, capas_perceptron)

#Grafico Error vs Iteracion
plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución del error")
plt.plot(np.arange(errores.shape[0]), errores, label = f"Perceptrón de capas { capas_perceptron } .")
plt.grid()
plt.legend()
plt.xlabel("Iteraciones")
plt.ylabel("Error")   

##Grafico Error vs Temperatura
TEMPERATURA_INICIAL = 40
ALFA = 0.95
T = np.array([TEMPERATURA_INICIAL])
for i in range(errores.shape[0]-1):
    T = np.append(T, T[-1]*ALFA)

plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución del error")

plt.plot( T,errores, label = f"Perceptrón de capas { capas_perceptron } .")
plt.grid()
plt.legend()
plt.xlabel("Temperatura")
plt.ylabel("Error")   
plt.xlim(max(T), min(T))
