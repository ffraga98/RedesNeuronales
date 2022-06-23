import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions import entradas, xor, capas_perceptron
from my_utils.functions_ej3 import sortear_pesos, error_al_modificar_deltaW, entrenar_perceptron_multicapa
import seaborn as sns
import pandas as pd

""" EJERCICIO 3

a. Implementar un perceptrón multicapa que aprenda la función lógica XOR de 2 
entradas y 4 entradas. Utilizando Backpropagation.

b. Muestre cómo evoluciona el error durante el entrenamiento.

"""
#Numero de entradas 
ENTRADAS = 2
#Capas ocultas
CAPAS_OCULTAS_PERCEPTRON = np.array([7])

x_input = entradas(ENTRADAS)
xor_output = xor(x_input)

#Capas de neuronas del perceptron
capas_perceptron = capas_perceptron(x_input, xor_output, CAPAS_OCULTAS_PERCEPTRON)


#Configuracion del ejercicio
W, errores = entrenar_perceptron_multicapa(x_input, xor_output, capas_perceptron)
plt.figure()
plt.rcParams["figure.figsize"] = (6,4)
plt.title(f"Evolución del error hasta ser menor a %0.3f" % errores[-1])
plt.plot(np.arange(errores.shape[0]), errores, label = f"Perceptrón de capas { capas_perceptron } .")
plt.grid()
plt.legend()
plt.xlabel("Iteraciones")
plt.ylabel("Error")       
#%%
"""
    c. Para una red entrenada en la función XOR de dos entradas, grafique el
    error en función de dos pesos cualesquiera de la red. 
    De ejemplos de mínimos y mesetas.
""" 
dw_lim = 4*2
deltas_w = np.linspace(-dw_lim,dw_lim, num = 100 )
color = sns.color_palette("RdYlGn",24)

#%% EJ 3c Modificar los arrays


capas, i, j = sortear_pesos(W)
#%%
#Primero lo hago con todos los patrones
error_dw = error_al_modificar_deltaW(deltas_w, W, x_input, xor_output, capas ,i, j)
df = pd.DataFrame( error_dw - errores[-1] )

plt.rcParams["figure.figsize"] = (8, 6)
fig, ax = plt.subplots(1)

plt.title(f"Modificando los pesos w{j[0]}{i[0]} y w{j[1]}{i[1]} de las capas {capas[0]} y {capas[1]}")
p1 = sns.heatmap(df, cmap = color, ax = ax)

ticks = np.arange(0,105,10)
ax.set_xticks( ticks)
ax.set_yticks( ticks )
labels = np.linspace(-dw_lim,dw_lim, num = ticks.size )
labels=np.array(np.round(labels,1).astype('str').tolist())
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_ylabel(r"$\Delta w_2$")
ax.set_xlabel(r"$\Delta w_1$")

# EJ 3d - lo hago con cada patron
# capas, i, j = sortear_pesos()
plt.rcParams["figure.figsize"] = (11, 9 )
fig, axs = plt.subplots(2,2)
axs = axs.flatten()
ticks = np.arange(0,105,10)
labels = np.linspace(-dw_lim,dw_lim, num = ticks.size )
labels=np.array(np.round(labels,2).astype('str').tolist())

for x,y,ax in zip(x_input, xor_output, axs):
        ax.set_title(f"Patron {x}")
        x = np.array([x])
        y = np.array([y])
        
        error_dw = error_al_modificar_deltaW(deltas_w, W, x, y, capas, i, j)
        df = pd.DataFrame( error_dw - errores[-1] )

        p = sns.heatmap(df, cmap = color, ax = ax)

        p.set_xticks( ticks)
        p.set_yticks( ticks )
        p.set_xticklabels(labels)
        p.set_yticklabels(labels)
