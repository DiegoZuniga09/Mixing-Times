#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplos de Cadenas de Markov
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






def graficar(simulacion, title, Polya = False):
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
        la Urna de Polya con la finalidad de cambiar el nombre del eje x.
        
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
    plt.plot(df["tiempo"], df["estados"], color="blue")
    plt.xlabel("Tiempo")
    if Polya:
        plt.ylabel("Proporción de bolas negras dentro de la urna")
    else:
        plt.ylabel("Estado")
        plt.xticks(df["tiempo"])  # Coloca marcas en el eje x en todos los valores de tiempo
        plt.yticks(range(int(min(simulacion)), int(max(simulacion)) + 1))  # Coloca marcas en el eje y en los valores enteros
    plt.title(titulo)
    
    plt.legend()
    plt.grid(True)  # Agrega una cuadrícula al gráfico
    plt.show()

def simulacion(pasos, matriz, xo):
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
    lista : list
        Una lista donde cada elemento es una sub-lista con el tiempo y el valor
        de la cadena en ese tiempo.

    """
    lista = list()
    lista.append(xo)
    
    for i in range(1,pasos):
        lista.append((np.random.choice(range(0,len(matriz)), p = matriz[lista[i-1],])))
    
    return lista


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

N = int(input("Ingrese la fortuna máxima: \n"))
p = float(input("Probabilidad de éxito: \n"))



matriz = matriz_transicion_ruina(p, N)
print("Matriz de transición: \n"
      f"{matriz}\n")


np.random.seed(160290)
xo = np.random.choice(len(matriz), p = np.full(len(matriz), 1/len(matriz)) )
print(f"Estado inicial dado por: {xo}\n")
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: \n"))
sim1 = simulacion(tiempos, matriz, xo)
graficar(sim1, "Simulación Ruina del Jugador")



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

c = int(input("Cuántos cupones son: "))

matriz = matriz_transicion_cupones(c)
print("Matriz de transición: ")
print(matriz)
np.random.seed(160290)
xo = np.random.choice(len(matriz), p = np.full(len(matriz), 1/len(matriz)) )
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
sim2 = simulacion(tiempos, matriz, xo)
graficar(sim2, "Simulación Coleccionista de Cupones")



"""
Urna de Polya
"""
def urna_polya(steps):
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

#La urna de Polya en esta cadena se modela como la proporcion
# al tiempo t
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))

sim3 = urna_polya(tiempos)
graficar(sim3, "Simulación Urna de Polya", True)



"""
Cadena de nacimientos y muertes
"""


#Básicamente puede modelarse con el código de una CM arbitraria
#pues en cada renglón (i) las probabilidades de ir de i -> i+1, i -> i,
# i -> i-1, no es la misma para toda i.
###Si sería necesario meter los valores para ver cómo se comporta 
# el proceso.




