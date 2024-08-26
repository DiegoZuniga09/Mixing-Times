#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:12:20 2024

@author: andrea
"""

import ejemplos_cadenas_markov as ECM

# Parámetros de la simulación
p = float(input("Ingrese la probabilidad de moverse hacia la derecha (entre 0 y 1): "))
r = float(input("Ingrese la probabilidad de quedarse en el mismo estado (entre 0 y 1): "))
q = 1-p-r

x0 = int(input("Ingrese el estado inicial de la caminata aleatoria en Z: "))
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
seed = int(input("Establece la semilla: "))

sim_caz = ECM.caminata_aleatoria_Z(tiempos, x0, seed, p, r, q)
ECM.graficar(sim_caz, "Simulación Caminata Aleatoria en Z")