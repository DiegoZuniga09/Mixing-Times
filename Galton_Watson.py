#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ejemplos_cadenas_markov as ECM

# Parámetros de la simulación
mu = float(input("Ingrese la media de individuos mu: "))
x0 = int(input("Ingrese el número de individuos inicial : "))
generaciones = int(input("Ingrese la cantidad de generaciones para graficar: "))
seed = int(input("Establece la semilla: "))

sim_gw = ECM.galton_watson(generaciones, x0, mu, seed)
ECM.graficar(sim_gw, "Simulación Modelo Galton-Watson", GW = True)
