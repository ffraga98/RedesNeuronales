import numpy as np
import matplotlib.pyplot as plt    
from myfunctions.utils2 import NEURONAS_LARGO,kohonen, circulo_unitario, cuadrado_unitario, inicializar_W

n_ciudades = 200
#%%


ciudades_circulo = circulo_unitario(n_ciudades)
W = inicializar_W(NEURONAS_LARGO)
W_recolectadas_circulo = kohonen(ciudades_circulo, W)

#%%
t = np.linspace(0,4,100)
t *= np.pi

aux_x = np.cos(t/2) 
aux_y = np.sin(t/2) 
fig, axes = plt.subplots(2,2)

for axs, ws in zip(axes, W_recolectadas_circulo):
    for ax, w in zip(axs, ws):
        ax.grid(True, which='both', linestyle = '--')
        ax.plot(aux_x,aux_y, "k")
        ax.scatter( ciudades_circulo[:,0], ciudades_circulo[:,1],  marker = 'x')
        ax.plot(w[:,0], w[:,1], "r", linewidth=0.7)
        ax.plot(w[:,0].transpose(), w[:,1].transpose(), "r",  linewidth=0.7)
    
axes[0,0].set_title("Inicial")
axes[0,1].set_title("Evolucion - 1")
axes[1,0].set_title("Evolucion - 2")
axes[1,1].set_title("Final")

#%%
ciudades = cuadrado_unitario(n_ciudades)
W = inicializar_W(NEURONAS_LARGO)
W_recolectadas_cuadrado = kohonen(ciudades, W)

#%%
t = np.linspace(-1,1,100)

fig, axes = plt.subplots(2,2)

for axs, ws in zip(axes, W_recolectadas_cuadrado):
    for ax, w in zip(axs, ws):
        ax.grid(True, which='both', linestyle = '--')
        
        ax.plot(t,np.ones(t.shape)*-1, "k")
        ax.plot(t,np.ones(t.shape)*1, "k")
        ax.plot(np.ones(t.shape)*-1,t, "k")
        ax.plot(np.ones(t.shape)*1,t, "k")
    
        ax.scatter( ciudades[:,0], ciudades[:,1],  marker = 'x')
        ax.plot(w[:,0], w[:,1], "r", linewidth=0.7)
    
axes[0,0].set_title("Inicial")
axes[0,1].set_title("Evolucion - 1")
axes[1,0].set_title("Evolucion - 2")
axes[1,1].set_title("Final")