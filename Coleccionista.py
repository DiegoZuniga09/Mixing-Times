#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:09:04 2024

@author: andrea
"""
import ejemplos_cadenas_markov as ECM
import numpy as np

c = int(input("Cuántos cupones son: "))

matriz_cupones = ECM.matriz_transicion_cupones(c)
print("Matriz de transición: ")
print(matriz_cupones)
x0 = int(input(f"Escoge el estado inicial entre 0 y {c}:\n"))
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
seed = int(input("Establece la semilla: "))
np.random.seed(seed) # 1007
sim_cupones = ECM.simulacion(tiempos, matriz_cupones, x0 )
ECM.graficar(sim_cupones, "Simulación Coleccionista de Cupones",Cupones = True)

