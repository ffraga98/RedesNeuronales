""" EJERCICIO 5
a- Encontrar un perceptrÃ³n multicapa que resuelva una XOR de 2 entradas con
 un algoritmo genÃ©tico. Graficar el fitness a lo largo del proceso de evoluciÃ³n
 
b- Como impacta en el aprendizaje la constante de mutaciÃ³n, la probabilidad del
cross-over y el tamaÃ±o de la poblacion.

"""
import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions import signo, entradas, xor, capas_perceptron
from my_utils.functions_ej5 import algoritmo_genetico, plot_generacion

#Numero de entradas 
ENTRADAS = 2
#Capas ocultas
CAPAS_OCULTAS_PERCEPTRON = np.array([7])

x_input = entradas(ENTRADAS)
xor_output = xor(x_input)
signo(x_input)
signo(xor_output)

#Capas de neuronas del perceptron
capas_perceptron = capas_perceptron(x_input, xor_output, CAPAS_OCULTAS_PERCEPTRON)

TAMANIO_POBLACION = 100
P_MUTACION = 0.6
P_CRUZA = 0.6
#%%
fit_generaciones_elite = algoritmo_genetico(x_input, xor_output, capas_perceptron, TAMANIO_POBLACION, P_MUTACION , P_CRUZA)
fit_generaciones = algoritmo_genetico(x_input, xor_output, capas_perceptron, TAMANIO_POBLACION, P_MUTACION , P_CRUZA, elite = False)

plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Comparacion de evolución del fitness")

x = np.arange(fit_generaciones_elite.shape[0])
plt.plot(x, fit_generaciones_elite, label = f"Con elite, generacion: {x[-1]}" )
x = np.arange(fit_generaciones.shape[0])
plt.plot(x, fit_generaciones, label = f"Sin elite, generacion: {x[-1]}" )


plt.grid()
plt.legend()
plt.xlabel("Generaciones")
plt.ylabel("Promedio Fitness")   
#%%

TAMANIO_POBLACION = 50
P_MUTACION = 0.3
p_cruza = np.linspace(0, 1, 20)

iteraciones = np.array([])
for pc in p_cruza:
    fit_generaciones = algoritmo_genetico(x_input, xor_output, capas_perceptron, TAMANIO_POBLACION, P_MUTACION , pc)
    iteraciones = np.append( iteraciones, fit_generaciones.shape[0])
plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución iteraciones por P(cruza)")

x = np.arange(fit_generaciones_elite.shape[0])
plt.plot(p_cruza, iteraciones, label = f"Tamaño: {TAMANIO_POBLACION} , p(mutacion): {P_MUTACION}" )

plt.grid()
plt.legend()
plt.ylabel("Iteraciones")
plt.xlabel("Prob. Cruza")   

#%%
TAMANIO_POBLACION = 50
P_CRUZA = 0.3
p_muta = np.linspace(0.1, 1, 20)

iteraciones = np.array([])
for pm in p_muta:
    fit_generaciones = algoritmo_genetico(x_input, xor_output, capas_perceptron, TAMANIO_POBLACION, pm , P_CRUZA)
    iteraciones = np.append( iteraciones, fit_generaciones.shape[0])
    
plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución iteraciones por P(mutacion)")

x = np.arange(fit_generaciones_elite.shape[0])
plt.plot(p_muta, iteraciones, label = f"Tamaño: {TAMANIO_POBLACION} , p(cruza): {P_CRUZA}" )

plt.grid()
plt.legend()
plt.ylabel("Iteraciones")
plt.xlabel("Prob. Mutacion")   


#%%
P_CRUZA = 0.6
P_MUTACION = 0.6
tamanio_p = np.arange(10, 106, 6)

iteraciones = np.array([])
for tm in tamanio_p:
    fit_generaciones = algoritmo_genetico(x_input, xor_output, capas_perceptron, tm, P_MUTACION , P_CRUZA)
    iteraciones = np.append( iteraciones, fit_generaciones.shape[0])
#%%
plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución iteraciones por tamaño")

x = np.arange(fit_generaciones_elite.shape[0])
plt.plot(tamanio_p, iteraciones, label = f"p(mutacion): {P_MUTACION} , p(cruza): {P_CRUZA}" )

plt.grid()
plt.legend()
plt.ylabel("Iteraciones")
plt.xlabel("Tamaño población")   

