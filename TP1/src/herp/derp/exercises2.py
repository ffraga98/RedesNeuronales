from herp.derp.functions import calcular_error, calculo_capacidad, calculo_neuronas_patrones
import numpy as np
import matplotlib.pyplot as plt

def ejercicio2_A():
    # Valores de la tabla a aproximar.
    probabilidadError = [0.001, 0.0036, 0.01, 0.05, 0.1]
    patrones_N = np.array([0.105, 0.138, 0.185, 0.37, 0.61])

    #Fijo la cantidad de imagenes, distinta para cada probabilidad de error.
    max_img = [22, 24, 30, 40, 60]
    paso = [1,1,2,3,5]
    capacidades = np.empty([0,5])
    #Itero sobre cada una de las probabilidades de error, etc.
    for error, capacidad_esperada, imgs, step in zip(probabilidadError, patrones_N, max_img, paso):
        print("Probabilidad de Error:", error)
        num_img = np.arange(4*step, imgs + step, step)
        
        n, num_img = calculo_neuronas_patrones(error, capacidad_esperada, num_img)
                    
        #Grafica de los patrones enseñados y la neuronas requeridas para 
        #obtener una probabilidad de error menor a la de la tabla.
        fig = plt.figure()
        capacidad, ord_origen = calculo_capacidad(n, num_img)
        polinomio = capacidad * n + ord_origen
        plt.plot(n, polinomio, label = str(capacidad) + " x + " + str(ord_origen))
        plt.scatter(n, num_img)
        plt.xlabel('Neuronas')
        plt.ylabel('Patrones enseñados')
        plt.title("Probabilidad de Error " + str(error))
        plt.legend()
        plt.grid(which = 'both', axis = 'both', linestyle = '--')

        capacidades = np.append(capacidades, capacidad)
        print(capacidad)
        

    
def ejercicio2_B():
    # Fijo la probabilidad de error
    probabilidadError = 0.001
    patrones_N = 0.105
    max_img = 8
    paso = 1
    
    capacidades = np.empty(0)
    #No recorro las covarianzas negativas porque es redundante.
    #No reccoro hasta covarianza = 1, por la demora computacional.
    covarianzas = np.arange(0,0.4,0.05)
    #Calculo las capacidades para cada covarianza.
    for cov in covarianzas:
        print("Covarianza: ", cov)
        num_img = np.arange(5, int(5 + max_img*(1 - cov) + paso), paso)
        n, num_img = calculo_neuronas_patrones(probabilidadError, patrones_N, num_img, cov)
        capacidad, ordenada =  calculo_capacidad(n, num_img)
        capacidades = np.append(capacidades, capacidad)
        print("Capacidad: ", capacidad)
        
    #Grafico de la capacidad en funcion a la capacidad calculada.
    fig = plt.figure()
    plt.plot(covarianzas, capacidades)
    plt.xlabel('Covarianza')
    plt.ylabel('Capacidad calculada')
    plt.title("Probabilidad de Error " + str(probabilidadError))
    plt.grid(which = 'both', axis = 'both', linestyle = '--')
    print(capacidades)




    
