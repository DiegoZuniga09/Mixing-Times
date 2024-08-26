#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:10:17 2024

@author: andrea
"""

import ejemplos_cadenas_markov as ECM
import numpy as np


N = int(input("Número de pelotas en total: "))

matriz_ehrenfest = ECM.matriz_transicion_ehrenfest(N)
print("Matriz de transición: \n"
      f"{matriz_ehrenfest}")

x0 = int(input(f"Escoge el estado inicial entre 0 y {N}:\n"))
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
seed = int(input("Establece la semilla: "))
np.random.seed(seed)#1007
sim_ehrenfest = ECM.simulacion(tiempos, matriz_ehrenfest, x0 )
ECM.graficar(sim_ehrenfest, "Simulación Urnas de Ehrenfest", Ehrenfest = True)
