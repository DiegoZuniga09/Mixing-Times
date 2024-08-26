#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:06:31 2024

@author: andrea
"""

import ejemplos_cadenas_markov as ECM
import numpy as np

N = int(input("Ingrese la fortuna máxima: \n"))
p = float(input("Probabilidad de éxito: \n"))



matriz_ruina = ECM.matriz_transicion_ruina(p, N)
print("Matriz de transición: \n"
      f"{matriz_ruina}\n")

seed = int(input("Establece la semilla: ")) #1007

np.random.seed(seed)
x0 = int(input(f"Escoge el estado inicial entre 0 y {N}:\n"))
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: \n"))
sim_ruina = ECM.simulacion(tiempos, matriz_ruina, x0)
ECM.graficar(sim_ruina, "Simulación Ruina del Jugador", Ruina=True)



