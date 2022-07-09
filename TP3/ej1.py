

"""
1. Hacer una red de Kohonen de 2 entradas que aprenda una distribución uniforme
dentro del círculo unitario. Mostrar el mapa de preservación de topología.
Probar con distribuciones uniformes dentro de otras figuras geométricas.
"""
import numpy as np
import matplotlib.pyplot as plt    
from myfunctions.utils import NEURONAS_ALTO,NEURONAS_ANCHO,kohonen, circulo_unitario, cuadrado_unitario, inicializar_W

max_patrones = 500
#%%
patrones_circulo = circulo_unitario(max_patrones)
W = inicializar_W(NEURONAS_ANCHO,NEURONAS_ALTO)
W_recolectadas_circulo = kohonen(patrones_circulo, W)

#%%
t = np.linspace(0,4,100)
t *= np.pi

aux_x = np.cos(t/2) 
aux_y = np.sin(t/2) 
fig, axs = plt.subplots(1,3)

for ax,w in zip(axs, W_recolectadas_circulo):
    ax.grid(True, which='both', linestyle = '--')
    ax.plot(aux_x,aux_y, "k")
    ax.scatter( patrones_circulo[:,0], patrones_circulo[:,1],  marker = 'x')
    ax.plot(w[:,:,0], w[:,:,1], "r", linewidth=0.7)
    ax.plot(w[:,:,0].transpose(), w[:,:,1].transpose(), "r",  linewidth=0.7)
    
axs[0].set_title("Inicial")
axs[1].set_title("Evolucion")
axs[2].set_title("Final")

#%%
patrones = cuadrado_unitario(max_patrones)
W = inicializar_W(NEURONAS_ANCHO,NEURONAS_ALTO)
W_recolectadas = kohonen(patrones, W)

#%%
t = np.linspace(-1,1,100)

fig, axs = plt.subplots(1,3)

for ax,w in zip(axs, W_recolectadas):
    ax.grid(True, which='both', linestyle = '--')
    
    ax.plot(t,np.ones(t.shape)*-1, "k")
    ax.plot(t,np.ones(t.shape)*1, "k")
    ax.plot(np.ones(t.shape)*-1,t, "k")
    ax.plot(np.ones(t.shape)*1,t, "k")

    ax.scatter( patrones[:,0], patrones[:,1],  marker = 'x')
    ax.plot(w[:,:,0], w[:,:,1], "r", linewidth=0.7)
    ax.plot(w[:,:,0].transpose(), w[:,:,1].transpose(), "r",  linewidth=0.7)
    
axs[0].set_title("Inicial")
axs[1].set_title("Evolucion")
axs[2].set_title("Final")

