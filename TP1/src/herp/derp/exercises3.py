from herp.derp.functions import olvidar_aprendizaje, calculo_neuronas_patrones,calculo_capacidad, reconstruct_syn, ley_hebb_gral,calcular_error_patron, generate_random_patterns #Necesito calcular el error y la capacidad, a medida que elimino interconexiones.
import numpy as np
import matplotlib.pyplot as plt
# ¿Que es lo que espero ver? Espero es que a medida que yo vaya eliminando valores de W,
#necesite mas neuronas o menos imagenes para lograr cumplir con el error.

#  El primer calculo es facil, puedo enseñarle una cantidad fija de patrones con las mismas neuronas
# e ir calculando como varia el error a medida que borro valores de la misma W.
def ejercicio3_A():
    #Voy a realizar el calculo para distinta cantidad de patrones enseñados.
    patrones = [10,20]
    N = 600

    porcentajes = np.arange(1001)/10
    fig = plt.figure()
    
    
    for num_patterns in patrones:
        #Crear los patrones pseudoaleatorios
        patterns = generate_random_patterns(num_patterns, N)
        #Entrenar
        W = ley_hebb_gral(patterns)
        
        errores = np.empty(0)
        for i in porcentajes:
            print("Porcentaje:", i)
            #Olvido parte de la matriz de sinapsis.
            W_olvidada = olvidar_aprendizaje(W, i)
            error = np.empty(0)
            #Calculo el error.
            for p in patterns:
                error = np.append(error,calcular_error_patron(p, reconstruct_syn(W_olvidada, p)))
            print("Error:", np.mean(error))
            errores = np.append(errores, np.mean(error))
        
        #Grafico el error vs porcentaje borrado.
        plt.plot(porcentajes,errores, label = "Patrones =" + str(num_patterns))
        plt.scatter(porcentajes[::100],errores[::100])
        
    plt.ylabel('Probabilidad de Error')
    plt.xlabel('Porcentaje olvidado')
    plt.title("Numero de Neuronas " + str(N))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')    
    plt.legend()


def ejercicio3_B():
    fig = plt.figure()
    #Fijo la probabilidad de error
    probabilidadError = 0.0036
    patrones_N =  0.138

    max_img = 5
    paso = 1
    
    capacidades = np.empty(0)
    # No calculo hasta porcentaje 100 por demora computacional.
    porcentajes = np.arange(91) 
    for i in porcentajes:
        print("Porcentaje de olvido:", i)
        num_img = np.arange(2, max_img + paso, paso)
        n, num_img = calculo_neuronas_patrones(probabilidadError, patrones_N, num_img, porcentaje_olvido = i)
        capacidad = calculo_capacidad(n, num_img)[0]
        print("Capacidad Estimada:", capacidad)
        capacidades = np.append(capacidades, capacidad)

    #Grafico la capacidad y el porcentaje de interconexiones eliminadas.
    plt.plot(porcentajes, capacidades)
    plt.scatter(porcentajes[::10],capacidades[::10])
        
    plt.xlabel('Porcentaje olvidado')
    plt.ylabel('Capacidad estimada')
    plt.title("Probabilidad de Error " + str(probabilidadError))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')    
    print(capacidades)