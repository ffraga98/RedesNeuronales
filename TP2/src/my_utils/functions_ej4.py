import numpy as np
import matplotlib.pyplot as plt
from my_utils.functions import capas_perceptron, inicializar_W
from my_utils.functions_ej3 import calcular_salidas, backdrop_propagation, error,calcular_salida_real

NUMERO_MUESTRAS_TRAIN = 1000
NUMERO_MUESTRAS_TEST = 500
ERROR_TARGET = 0.01 * NUMERO_MUESTRAS_TEST / NUMERO_MUESTRAS_TRAIN
MAX_ITERACIONES = 5e4
ETA = 0.005

def entrenar_perceptron_multicapa(capas_ocultas, patrones_por_batch ):
    
    x = np.linspace(0, 2* np.pi, NUMERO_MUESTRAS_TEST)
    y = np.linspace(0, 2* np.pi, NUMERO_MUESTRAS_TEST)
    z = np.linspace(-1 , 1, NUMERO_MUESTRAS_TEST)

    set_testeo = generar_set_muestras_aleatoria(x,y,z)

    x = np.linspace(0, 2* np.pi, NUMERO_MUESTRAS_TRAIN)
    y = np.linspace(0, 2* np.pi, NUMERO_MUESTRAS_TRAIN)
    z = np.linspace(-1 , 1, NUMERO_MUESTRAS_TRAIN)

    set_entrenamiento = generar_set_muestras_aleatoria(x,y,z)
    
    capas_perceptron_ = capas_perceptron(set_entrenamiento, np.zeros([1,1]), capas_ocultas)

    W = inicializar_W(capas_perceptron_)
    error_ = 1
    errores_epoca = np.array([]) 
    errores_iteracion = np.array([]) 
    

    while error_ > ERROR_TARGET and len(errores_epoca) < MAX_ITERACIONES:
        
        np.random.shuffle(set_entrenamiento)
        set_ = set_entrenamiento.tolist()
                     
        salidas_deseadas = np.array([])
        salidas_reales = np.array([])
        while( len(set_) ):
            i=0
            sum_deltas = 0
            while( i < patrones_por_batch and len(set_)):
                entrada = set_.pop()
                salida_deseada = funcion_xyz(entrada)/3
                salidas_capas = calcular_salidas(W, entrada)
                deltas = backdrop_propagation(ETA, salidas_capas, W, salida_deseada, entrada)
                sum_deltas += deltas   
                i += 1
                salida_r = calcular_salida_real(W, np.array([entrada]))
                errores_iteracion = np.append(errores_iteracion, error(np.array([salida_deseada]), salida_r))
                
            W += sum_deltas
            
        errores = np.array([])
        for entrada in set_testeo:
            salida_deseada = np.array([funcion_xyz(entrada)])/3
            salida_r = calcular_salida_real(W, np.array([entrada]))
            errores = np.append(errores, error(salida_deseada, salida_r))
            salidas_deseadas = np.append(salidas_deseadas, salida_deseada)
            salidas_reales = np.append(salidas_reales, salida_r)
        

        error_ = np.mean(errores)
        errores_epoca = np.append(errores_epoca, error_)
        print(f"Epoca: {len(errores_epoca)}, Error: {error_}")

    graficar_regresion(salidas_deseadas, salidas_reales)
    return errores_epoca, errores_iteracion


def generar_set_muestras_aleatoria(x,y,z):
    set_ = np.zeros([x.size, 3])

    np.random.shuffle(x)
    np.random.shuffle(y)
    np.random.shuffle(z)

    for index, s in enumerate(set_):
        set_[ index ][0] = x[index]
        set_[ index ][1] = y[index]
        set_[ index ][2] = z[index]
        
    return set_


def funcion_xyz(vector_xyz):
    x = vector_xyz[0]
    y = vector_xyz[1]
    z = vector_xyz[2]    
    return np.sin(x) + np.cos(y) + z

def plot_errores(e_epoca, e_entrenamiento, patrones_por_batch):
    #Mejorar, debería agregar el error de cada patron
    #y marcar en el eje x las iteraciones, dependiendo del tamaño del batch
    plt.figure()
    plt.rcParams["figure.figsize"] = (6,4)
    plt.title(f"Entrenamiento, Emin = %.4f" % e_entrenamiento[-1])
    plt.plot(np.linspace(1,len(e_epoca), len(e_entrenamiento)), e_entrenamiento, label = f"Minibatch { patrones_por_batch }")
    plt.grid()
    plt.legend()
    plt.xlabel("Iteraciones")
    plt.ylabel("Error")   
    
    
    plt.figure()
    plt.rcParams["figure.figsize"] = (6,4)
    plt.title(f"Testeo, Emin = %.4f" % e_epoca[-1])
    plt.plot(np.linspace(1,len(e_epoca), len(e_epoca)), e_epoca, label = f"Minibatch { patrones_por_batch }")
    plt.grid()
    plt.legend()
    plt.xlabel("Iteraciones")
    plt.ylabel("Error")   
    
    

def graficar_regresion(salida_deseada, salida_real):
    fig = plt.figure()
    plt.scatter( salida_deseada, salida_real, s = 0.25, label = "Datos")
    #Grafico una aproximación a las muestras.
    coef = np.polynomial.polynomial.Polynomial.fit( salida_deseada , salida_real , 2).convert().coef

    
    T = np.arange(min(salida_deseada),max(salida_deseada), 0.1)
    polinomio = coef[1] * T + coef[0]
    plt.plot(T, polinomio, label = "Aproximación recta")  
    plt.xlim(min(salida_deseada), max(salida_deseada))      
    plt.legend()
    plt.ylabel("Salida real")
    plt.xlabel("Salida calculada")
    plt.title("Recta regresion" )
    plt.grid(which = 'both', axis = 'both', linestyle = '--')