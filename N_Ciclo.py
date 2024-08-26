#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:59:27 2024

@author: andrea
"""


import ejemplos_cadenas_markov as ECM
import numpy as np
import funciones_CM as FCM

N = int(input("Ingrese el tamaño del Ciclo: "))

p = float(input("Ingrese la probabilidad de moverse hacia la derecha (entre 0 y 1): "))
q = float(input("Ingrese la probabilidad de moverse hacia la izquierda (entre 0 y 1): "))
r = 1-p-q

x0 = int(input("Ingrese la posición dentro del ciclo \n"
               f"(de 0 a {N}): "))
tiempos = int(input("Ingrese el número de tiempos: "))

matriz_nciclo = ECM.matriz_transicion_nciclo(N)
print("Matriz de transición: \n"
      f"{matriz_nciclo}")

# Representación
FCM.visualizar(matriz_nciclo)

# Realizar la simulación del modelo de Wright-Fisher
seed = int(input("Establece la semilla: "))
np.random.seed(seed) #1007
sim_nciclo = ECM.simulacion(tiempos, matriz_nciclo, x0)

# Graficar los resultados
ECM.graficar(sim_nciclo, "Simulación de la Caminata Aleatoria en el N-Ciclo")
