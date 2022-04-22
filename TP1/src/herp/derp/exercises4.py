import numpy as np
import matplotlib.pyplot as plt
from herp.derp.functions import  actualizar, generate_vector_spines, matriz_conexion_dipolos

#MODELO ISING.
    
def ejercicio4_1D(N):
    #Creo un vector pseudoaleatorio de 1 y -1.
    S = generate_vector_spines( 1, N )
    filas = 1
        

    #Creacion de W.
    W = matriz_conexion_dipolos(filas, N)
    #Creo un vector de temperaturas con un paso de 0.05
    T = np.arange(6, 0, -0.05)
    
    s_mean = np.empty([0, T.shape[0]])
    indices = np.arange( N )
    #Recorro el vector de temperaturas
    for t in T:    
        print("Temperatura:", t)
        for i in range(10):
            np.random.shuffle(indices)
            for ix in indices:  
                new_S = np.copy(S)
                new_S[0,ix] *= -1
                S = actualizar(S, new_S, W, t )   
    
        s_mean = np.append(s_mean, np.mean(S))
            
    fig = plt.figure()
    plt.scatter(T, s_mean, label = "Datos")
    
    #Grafico una aproximación a las muestras.
    coef = np.polynomial.polynomial.Polynomial.fit( T , s_mean , 3).convert().coef
    polinomio = coef[3] * T**3 + coef[2] * T**2 + coef[1] * T**1 + coef[0]
        
    plt.plot(T, polinomio, label = "Aproximación polinomial")  
    plt.xlim(max(T), min(T))      
    plt.legend()
    plt.ylabel("<Spines>")
    plt.xlabel("Temperatura")
    plt.title("Numero total de spines " + str(N) + ", Dimension: 1D")
    plt.grid(which = 'both', axis = 'both', linestyle = '--')

def ejercicio4_2D(N):
    #Creo un vector pseudoaleatorio de 1 y -1.
    #Genera una matriz cuadrada, aunque el modelo de Ising es valido para cualquier matriz nxm.
    S = generate_vector_spines( int(np.sqrt(N)) , int(np.sqrt(N)) )
    filas = int(np.sqrt(N))
    N = int(np.sqrt(N))**2 #Parece redundante pero 
    # al castear a int las filas y columnas, la cantidad total de neuronas cambian,
    #Creacion de W.
    W = matriz_conexion_dipolos(filas, N)
    #Creo un vector de temperaturas con un paso de 0.05
    T = np.arange(6, 0, -0.05)
    
    s_mean = np.empty([0, T.shape[0]])
    indices = np.arange( N )
    #Recorro el vector de temperaturas
    for t in T:    
        print("Temperatura:", t)
        for i in range(10):
            np.random.shuffle(indices)
            for ix in indices:  
                new_S = np.copy(S)
                new_S[0,ix] *= -1
                S = actualizar(S, new_S, W, t )   
    
        s_mean = np.append(s_mean, np.mean(S))
            
    fig = plt.figure()
    plt.scatter(T, s_mean, label = "Datos")

    #Grafico una aproximación a las muestras.
    coef = np.polynomial.polynomial.Polynomial.fit( T , s_mean , 6).convert().coef
    polinomio = coef[6] * T**6 + coef[5] * T**5 + coef[4] * T**4 + coef[3] * T**3 + coef[2] * T**2 + coef[1] * T**1 + coef[0]
        
    plt.plot(T, polinomio, label = "Aproximación polinomial")  
    plt.xlim(max(T), min(T))      
    plt.legend()
    plt.ylabel("<Spines>")
    plt.xlabel("Temperatura")
    plt.title("Numero total de spines " + str(N) + ", Dimension: 2D" )
    plt.grid(which = 'both', axis = 'both', linestyle = '--')


        
        
        