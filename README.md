# Trabajos prácticos - FIUBA - [86.54] - Redes Neuronales

| [TP1](#trabajo-practico-n1-red-hopfield-82) | [TP2](#trabajo-practico-n2-perceptrón) | [TP3](#trabajo-practico-n3-kohonen) |
|---------------------------------------------|----------------------------------------|-------------------------------------|

## [Trabajo Practico N°1: Red Hopfield '82](https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/TP1_FRAGA_102369.pdf)

### Ejercicio 1
- Entrenamiento de una red de Hopfield '82 con un conjunto de imagenes binarias. 
- Evolución de la red inicializando de imagenes alteradas.
- Estados espurios.

<p align="center">
  <img alt="Reconstruccion Anonymous" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej1b/v.svg" width=30%>
  <img alt="Estados espurios" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej1d/aprendiendo_todas.svg" width=30%>
<p/>

### Ejercicio 2
- Cálculo estadístico de la capacidad de una red de Hopfield '82.
- Visualización de la dependencia de la capacidad en función de la correlación.

<p align="center">
  <img alt="Capacidad Probabilidad de Error 0.01" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej2a/pe_01.svg" width=30%>
  <img alt="Capacidad Probabilidad de Error 0.1" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej2a/pe_1.svg" width=30%>
<p/>

### Ejercicio 3
- Representación del error y la capacidad en función de sinapsis elimindas.

<p align="center">
  <img alt="Error vs sinapsis eliminadas 1" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej3a/error_porcentaje.svg" width=30%>
  <img alt="Error vs sinapsis eliminadas 2" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej3b/capacidad_olvido.svg" width=30%>
<p/>

### Ejercicio 4
- Modelo Ising en una y dos dimensiones.
<p align="center">
  <img alt="Ising 1D" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej4/1D.svg" width=30%>
  <img alt="Ising 1D" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP1/imagenesEjercicio/Ej4/2D.svg" width=30%>
<p/>

## [Trabajo Practico N°2: Perceptrón](https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/TP2_FRAGA_102369.pdf)

### Ejercicio 1

- Implementación de un preceptrón simple. Aprendizaje de las función lógicas de la $\mathbf{AND}$ y la $\mathbf{OR}$ ( 2 y 4 entradas).
- Visualización de *recta discriminadora*.

<p align="center">
  <img alt="Resultados 2 Entradas" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej1/pesos2entradas.png" width=35%>
  <img alt="Recta Discriminadora" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej1/rectasDiscriminatorias.png" width=30%>
<p/>

### Ejercicio 2

- Capacidad en función del número de patrones enseñados.

<p align="center">
  <img alt="Capacidad" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej2/capacidad.png" width=35%>
<p/>

### Ejercicio 3 

- Perceptrón multicapa. Algoritmos de Backpropagation.
- Evolución del error durante el entrenamiento.
- Visualización del error al variar dos pesos cualesquiera de la red.

<p align="center">
  <img alt="error variando dos pesos" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej3/error2.png" width=45%>
  <img alt="Mapa de calor error" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej3/heatmap_total.png" width=30%>
<p/>

### Ejercicio 4

- Aprendizaje de la función $f(x,y,z) = sin(x) + cos(y) + z$ donde  $x,y \in [0,2\pi]$ y $z \in [-1,1]$.
- Análisis del efecto que tiene el tamaño del *minibatch* y la constante de aprendizaje sobre el número de iteraciones y el tiempo de entrenamiento.

<p align="center">
  <img alt="recta de regresion" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej4/minibatch100_regresion.png" width=30%>
  <img alt="error entrenamiento" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej4/minibatch100_train.png" width=30%>
  <img alt="error pruebas" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej4/minibatch100_test.png" width=30%>
<p/>


### Ejercicio 5

- Perceptrón multicapa con algoritmo genético para una $\mathbf{XOR}$ de dos enradas. 
- Impacto de la constante de *mutación*, probabilidad de *crossover* y el tamaño de la población.

<p align="center">
  <img alt="Alg. Genetico - Cruza" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej5/t100pc3pm9.png" width=30%>
  <img alt="Alg. Genetico - Crossover" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej5/t100pc9pm3.png" width=30%>
  <img alt="Alg. Genetico - Tamanio" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej5/t200pc6pm6.png" width=30%>
<p/>

### Ejercicio 6

- Perceptrón multicapa del anterior ejercicio resolviendo con *simmulated annealing*.

<p align="center">
  <img alt="Simmulated Annealing - temperatura" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej6/temperatura.png" width=30%>
  <img alt="Simmulated Annealing - iteraciones" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP2/img/ej6/iteraciones.png" width=30%>
<p/>


## [Trabajo Practico N°3: Kohonen](https://github.com/ffraga98/RedesNeuronales/blob/main/TP3/TP3_FRAGA_FERNANDO.pdf)

### Ejercicio 1

- Construcción de una red de Kohonen de 2 entradas que aprenda distrubuciones uniformes de distintas formas. 

<p align="center">
  <img alt="Kohonen Circulo" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP3/img/ej1/circulo.png" width=45%>
  <img alt="Kohonen Cuadrado" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP3/img/ej1/cuadrado.png" width=53.5%>
<p/>

### Ejercicio 2

- Resolución de "Traveling Salesman Problem" para 200 ciudades.

<p align="center">
  <img alt="Salesman Circulo" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP3/img/ej2/mapa_circulo.png" width=33.7%>
  <img alt="Salesman Cuadrado" src="https://github.com/ffraga98/RedesNeuronales/blob/main/TP3/img/ej2/mapa_cuadrado.png" width=36%>
<p/>
