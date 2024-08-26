#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplos de Cadenas de Markov
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






def graficar(simulacion, title, Polya = False,Ruina=False,Cupones=False,
             WF =False, Ehrenfest = False):
    """
    Función que graficará la simulación de la cadena de Markov.

    Parameters
    ----------
    simulacion : list
        Lista que indica el estado en cada tiempo de la simulación 
        de tu cadena.
    title: str
        String que servirá para establecer el título de la simulación
    Polya: Boolean
        Booleano que servirá para indicar si se trata del ejemplo de 
        la Urna de Polya con la finalidad de cambiar el nombre de los ejes. 
    Ruina: Boolean
        Booleano que servirá para indicar si se trata del ejemplo de 
        la Ruina del Jugador con la finalidad de cambiar el nombre de los ejes.     
    Cupones: Boolean
        Booleano que servirá para indicar si se trata del ejemplo del 
        Coleccionista de Cupones con la finalidad de cambiar el nombre de los 
        ejes. 
    WF: Boolean
        Booleano que servirá para indicar si se trata del ejemplo del 
        Modelo Wright-Fisher con la finalidad de cambiar el nombre de los ejes. 
    Ehrenfest: Boolean
        Booleano que servirá para indicar si se trata del ejemplo de 
        las Urnas de Ehrenfest con la finalidad de cambiar el nombre de 
        los ejes. 
    Returns
    -------
    None.

    """
    titulo = str(title)
    
    df = pd.DataFrame({
        "estados": simulacion,  
        "tiempo": range(1, len(simulacion) + 1)
    })
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(df["tiempo"], df["estados"], color="black")
    plt.plot(df["tiempo"], df["estados"], marker='o', linestyle='-')
    if WF:
        plt.xlabel("Generación")
    elif Ruina:
        plt.xlabel("Juego")
    elif Cupones:
        plt.xlabel("Cupones Obtenidos")
    else:
        plt.xlabel("Tiempo")
        
    if Polya:
        pass
    else:
        max_time = len(simulacion)
        step_x = max(1, max_time // 10)  # Determina un paso razonable para las marcas en el eje x
        plt.xticks(range(0, max_time + 1, step_x))  # Coloca marcas cada 'step_x' unidades        
        
        min_estado = int(min(simulacion))
        max_estado = int(max(simulacion))
        step_y = max(1, (max_estado - min_estado) // 10)  # Determina un paso razonable para las marcas en el eje y
        plt.yticks(range(min_estado, max_estado + 1 , step_y))  # Coloca marcas cada 'step_y' unidades

    if WF:
        plt.ylabel("Cantidad de Alelos A")
    elif Ruina:
        plt.ylabel("Fortuna")
    elif Cupones:
        plt.ylabel("Cupones Diferentes")
    elif Ehrenfest:
        plt.ylabel("Pelotas en Urna A")
    elif Polya:
        plt.ylabel("Proporción de bolas negras dentro de la urna")
    else:
        plt.ylabel("Estado")
    plt.title(titulo)
    
    plt.legend()
    plt.grid(True)  # Agrega una cuadrícula al gráfico
    plt.show()


def simulacion(pasos, matriz, x0):
    """
    Función que se utilizará para simular los ejemplos de cadenas de Markov.

    Parameters
    ----------
    pasos : int
        Indica cuántos pasos simulará la cadena de Markov.
    matriz : np.array
        Matriz de probabilidades de transición.
    xo : int
        Estado inicial con el que se iniciará la simulación.

    Returns
    -------
    simulacion: list
        Una lista donde cada elemento es una sub-lista con el tiempo y el valor
        de la cadena en ese tiempo.

    """
    simulacion = [x0]
    
    for i in range(1,pasos):
        simulacion.append((np.random.choice(range(0,len(matriz)), 
                                       p = matriz[simulacion[i-1],])))
    
    return simulacion


"""
Ruina del jugador
"""


def matriz_transicion_ruina(probabilidad_exito, max_fortuna):
    """
    Función que proporcionará la matriz de transición del ejemplo de la 
    Ruina del Jugador

    Parameters
    ----------
    probabilidad_exito : float
        Es la probabilidad de éxito del jugador.
    max_fortuna : int
        Es la cantidad máxima de fortuna que puede adquirir el jugador.

    Returns
    -------
    matriz : np.array
        Matriz de probabilidades de transición.

    """
    matriz = np.zeros((max_fortuna + 1, max_fortuna + 1))

    for i in range(max_fortuna + 1):
        if i < max_fortuna:
            matriz[i, i + 1] = probabilidad_exito
        if i > 0:
            matriz[i, i - 1] = 1 - probabilidad_exito

    matriz[0, 0] = 1 
    matriz[0,1] = 0
    
    matriz[max_fortuna, max_fortuna-1] = 0
    matriz[max_fortuna, max_fortuna] = 1

    return matriz




"""
Coleccionista de cupones
"""

def matriz_transicion_cupones(cupones):
    """
    Función que proporcionará la matriz de transición del ejemplo del 
    Coleccionista de Cupones

    Parameters
    ----------
    cupones : int
        Número de cupones dentro de la colección.

    Returns
    -------
    matriz : np.array
        Matriz de probabilidades de transición.

    """
    matriz = np.zeros((cupones + 1, cupones + 1))

    for i in range(cupones):
        if i < cupones:
            matriz[i, i] = i/cupones
            matriz[i, i + 1] = (cupones-i)/cupones 

    matriz[cupones, cupones] = 1

    return matriz



"""
Las Urnas de Ehrenfest
"""
def matriz_transicion_ehrenfest(N):
    """
    Función que construye la matriz de transición para el modelo de urnas de Ehrenfest.

    Parameters
    ----------
    N : int
        Número total de bolas.

    Returns
    -------
    matriz : np.array
        Matriz de probabilidades de transición.
    """
    matriz = np.zeros((N + 1, N + 1))
    
    for i in range(N + 1):
        if i > 0:
            matriz[i, i - 1] = i / N  # Probabilidad de mover una pelota de la primera urna a la segunda urna
        if i < N:
            matriz[i, i + 1] = (N - i) / N  # Probabilidad de mover una pelota de la segunda urna a la primera urna

    return matriz





"""
Urna de Polya
"""
def urna_polya(steps, seed):
    """
    Función que proporcionará la simulación del
    ejemplo de la Urna de Polya.

    Parameters
    ----------
    steps : int
        Número de pasos que la cadena simulará.

    Returns
    -------
    proporcion : list
        Lista que contiene los tiempos y la proporción de la Urna.

    """
    y = 1 #el numero de bolas negras que tengo en cada tiempo
    x = y/2 #la proporcion de bolas negras en cada tiempo
    proporcion = [x]
    pasos = int(steps)
    np.random.seed(seed)
    
    for i in range(pasos):
        t = np.random.choice([1,0], size = 1, p = [y/(i+2), 1-y/(i+2)])
        
        if t == 0:
    
            x = y/(i+3)
            proporcion.append(x)
            
        else:
            y+=1
            x = y/(i+3)
            proporcion.append(x)
    return proporcion




"""
Caminata Aleatoria en Z
"""

def caminata_aleatoria_Z(pasos, x0, seed = 1007, p=0.25, r=0.5, q=0.25):
    """
    Función que simula una caminata aleatoria lazy en los enteros Z.

    Parameters
    ----------
    pasos : int
        Número de pasos de la caminata aleatoria.
    p : float
        Probabilidad de moverse hacia la derecha (por defecto 0.25).
    q : float
        Probabilidad de quedarse en el mismo estado (por defecto 0.5).
    x0: int
        Estado inicial de la caminata aleatoria
    Returns
    -------
    posiciones : list
        Lista con las posiciones en cada tiempo de la caminata.
    """
     

    # Inicialización
    posiciones = [x0]  # Posición inicial en x0
    np.random.seed(seed)
    for _ in range(pasos):
        movimiento = np.random.choice([1, 0, -1], p=[p, r, q])
        nueva_posicion = posiciones[-1] + movimiento
        posiciones.append(nueva_posicion)
    
    return posiciones




"""
Modelo Wright-Fisher
"""

from scipy.stats import binom

def matriz_transicion_wright_fisher(N):
    """
    Función que construye la matriz de transición para el modelo de Wright-Fisher.

    Parameters
    ----------
    N : int
        Tamaño de la población.

    Returns
    -------
    matriz : np.array
        Matriz de probabilidades de transición.
    """
    matriz = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            matriz[i, j] = binom.pmf(j, N, i / N)

    return matriz


"""
Caminata Aleatoria en el N-ciclo
"""

def matriz_transicion_nciclo(N, p=0.5, r=0, q=0.5):
    """
    Función que proporciona la matriz de transición para una caminata aleatoria en un N-ciclo.

    Parameters
    ----------
    N : int
        Número de estados en el ciclo.
    p : float
        Probabilidad de moverse hacia la derecha.
    r : float
        Probabilidad de quedarse en el mismo estado.
    q : float
        Probabilidad de moverse hacia la izquierda.

    Returns
    -------
    matriz : np.array
        Matriz de probabilidades de transición.
    """
    matriz = np.zeros((N, N))

    for i in range(N):
        matriz[i, i] = r
        matriz[i, (i + 1) % N] = p
        matriz[i, (i - 1) % N] = q

    return matriz




