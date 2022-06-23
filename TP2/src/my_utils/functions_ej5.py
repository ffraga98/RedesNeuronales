import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions_ej3 import calcular_salida_real,inicializar_W

def fitness(error):
    return 1 - error/4

def fitness_poblacion(poblacion, x_input, xor_output):
    fit_poblacion = np.array([])
    for w in poblacion:
        dif = (xor_output.flatten() - calcular_salida_real(w, x_input))
        error_ = np.sum( dif**2 ) / x_input.shape[0]
        fit = fitness(error_)
        fit_poblacion = np.append( fit_poblacion , fit )
        
    return fit_poblacion

def cruzar( habitante1, habitante2):
    for w1,w2 in zip(habitante1, habitante2):
        swap_col( w1, w2)

def swap_col(w1, w2):
        col = np.random.randint(0, w1.shape[1])
        aux = w1
        w1[:,col] = w2[:, col]
        w2[:,col] = aux[:, col]

def mutar( habitante ):
    for w in habitante:
        w += np.random.normal(0,0.5, w.shape)

def inicializar_poblacion(tamanio, capas_perceptron):
    poblacion = np.zeros([tamanio], dtype= object)
    for i in range(tamanio):
        W = inicializar_W(capas_perceptron)
        habitante = np.array([0,0], dtype = object)
    
        for index, w in enumerate(W):
            habitante[ index ] = w
    
        poblacion[i] = habitante
    
    return poblacion

def algoritmo_genetico(x, y, capas_perceptron, tamanio_poblacion, p_mutacion, p_cruza, elite = True):
    poblacion = inicializar_poblacion(tamanio_poblacion, capas_perceptron)
    
    generacion = -1
    fprom = 0
    fit_generaciones = np.array([])
    while fprom < 0.9:
        generacion += 1
        
        ## Calcular el fitness de la poblacion
        fit_poblacion = fitness_poblacion(poblacion, x, y)  
        
        ##Condicion de corte
        fprom = np.mean(fit_poblacion)
        fit_generaciones = np.append(fit_generaciones, fprom)
        
        print(f"   Fitness promedio {fprom}")
        
        ## Reproducir
        probabilidad_rep = fit_poblacion / sum( fit_poblacion )
            
        nuevo_tamanio = 0
        nueva_poblacion = np.zeros( tamanio_poblacion, dtype = object)
        
        #Obtener el Elite
        index_elite = np.argmax( fit_poblacion )
        elite_0 = np.copy( poblacion[ index_elite ][0] )
        elite_1 = np.copy( poblacion[ index_elite ][1] )
       
        
        while( nuevo_tamanio < tamanio_poblacion):
            hab = np.random.randint(0,tamanio_poblacion)
            p = probabilidad_rep[hab]
            if( np.random.binomial(1,p) ):
                nueva_poblacion[ nuevo_tamanio ] = np.copy(poblacion[ hab ])
                nuevo_tamanio +=1
                
        #Supervivencia del Elite.
        if(elite):
            nueva_poblacion[ 0 ][0] = elite_0
            nueva_poblacion[ 0 ][1] = elite_1
        
        ## Cruzar
        num_cruza = np.arange(0, tamanio_poblacion, 1)
        np.random.shuffle( num_cruza )
        num_cruza = list( num_cruza )
        while( len(num_cruza)):
            hab1 = num_cruza.pop()
            hab2 = num_cruza.pop()
            if( np.random.rand() < p_cruza ):
                cruzar( nueva_poblacion[ hab1 ], nueva_poblacion[ hab2 ] )
    
    
    
        ## Mutar
        num_mutantes = np.arange(0, tamanio_poblacion, 1)
        num_mutantes = list( num_mutantes )
        
        while( len(num_mutantes)):
            hab = num_mutantes.pop()
            if( np.random.rand() < p_mutacion):
                mutar( nueva_poblacion[ hab ] )
    
        
        for w_vieja, w_nueva in zip(poblacion,nueva_poblacion):
            w_vieja[0] = np.copy(w_nueva[0])
            w_vieja[1] = np.copy(w_nueva[1])
            
        
    
        print(f"Generacion {generacion}", end = "")
        
    return fit_generaciones

def plot_generacion(fit_generaciones, tamanio, p_cruza, p_mutacion):
    plt.figure()
    plt.rcParams["figure.figsize"] = (6,4)
    plt.title(f"Fitness de la poblacion de tamaño {tamanio} \n con P(cruza) = {p_cruza} y P(mutacion) = {p_mutacion}")

    x = np.arange(fit_generaciones.shape[0])
    plt.plot(x, fit_generaciones, label = f"Generacion: {x[-1]}" )


    plt.grid()
    plt.legend()
    plt.xlabel("Generaciones")
    plt.ylabel("Promedio Fitness")   
