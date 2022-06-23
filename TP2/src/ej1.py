import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions_ej1 import entrenar_perceptron_simple
from my_utils.functions import and_input, or_input, xor, entradas, signo

""" EJERCICIO 1
Implementar un perceptrón simple que aprenda la función lógica AND de 2 y de 4
entradas.Lo mismo para la funcion OR. Para el caso de DOS dimensiones,
grafique la recta discriminadora y todos los vectores de entrada de la red.
"""

# Cada fila es un vector de entrada. Ordenados como x0 (bias), x1, x2

x_input2 = entradas(2)
x_input4 = entradas(4)

and_output2 = and_input( x_input2 )
and_output4 = and_input( x_input4 )

or_output2 = or_input(x_input2)
or_output4 = or_input(x_input4)

xor_output = xor( x_input2 )

signo(x_input2)
signo(x_input4)
print("")
print(" AND - 2 entradas")
w_final_AND, aprendido = entrenar_perceptron_simple( x_input2, and_output2)

print(" OR - 2 entradas")
w_final_OR, aprendido = entrenar_perceptron_simple( x_input2, or_output2)

print(" XOR - 2 entradas")
w_final_XOR, aprendido = entrenar_perceptron_simple( x_input2, xor_output)

print(" AND - 4 entradas")
entrenar_perceptron_simple( x_input4, and_output4)

print(" OR - 4 entradas")
entrenar_perceptron_simple( x_input4, or_output4)


#GRAFICOS

entradas = x_input2.transpose()
fig, ax = plt.subplots(1,3)


x = np.arange(-1.5, 1.6 , 0.1)

w_final_AND = w_final_AND.flatten()

ax[0].scatter(entradas[1][:3], entradas[0][:3], color = 'r')
ax[0].scatter(entradas[1][-1], entradas[0][-1], color = 'b')
ax[0].plot( x, - w_final_AND[1]/w_final_AND[2] * x - w_final_AND[0]/w_final_AND[2]  )

ax[0].set_title("AND")

w_final_OR = w_final_OR.flatten()


ax[1].scatter(entradas[1][0], entradas[0][0], color = 'r')
ax[1].scatter(entradas[1][1:], entradas[0][1:], color = 'b')
ax[1].plot( x, - w_final_OR[1]/w_final_OR[2] * x - w_final_OR[0]/w_final_OR[2]  )
ax[1].set_title("OR")


w_final_XOR = w_final_XOR.flatten()

ax[2].scatter( [1,-1], [-1,1], color = 'b')
ax[2].scatter( [-1,1], [-1,1], color = 'r')
ax[2].plot( x, - w_final_XOR[1]/w_final_XOR[2] * x - w_final_XOR[0]/w_final_XOR[2]  )
ax[2].set_title("XOR")


for a in ax:
    a.set_xticks(np.arange(-1,2,2))
    a.set_yticks(np.arange(-1, 2,2))
    a.grid(color = "0.8", linestyle = "--")
    a.set_xlim([-1.2, 1.2])
    a.set_ylim([-1.2, 1.2])

