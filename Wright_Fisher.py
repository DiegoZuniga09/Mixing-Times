#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:13:40 2024

@author: andrea
"""
import ejemplos_cadenas_markov as ECM
import numpy as np

N = int(input("Ingrese el tamaño de la población: "))
x0 = int(input("Ingrese el número inicial de individuos del alelo A: "))
generaciones = int(input("Ingrese el número de generaciones: "))

matriz_wf = ECM.matriz_transicion_wright_fisher(N)
print("Matriz de transición: \n"
      f"{matriz_wf}")

# Realizar la simulación del modelo de Wright-Fisher
seed = int(input("Establece la semilla: "))
np.random.seed(seed) #1007
sim_wf = ECM.simulacion(generaciones, matriz_wf, x0)

# Graficar los resultados
ECM.graficar(sim_wf, "Simulación de Wright Fisher", WF = True)

