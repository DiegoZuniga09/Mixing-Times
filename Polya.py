#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:11:25 2024

@author: andrea
"""

import ejemplos_cadenas_markov as ECM


#La urna de Polya en esta cadena se modela como la proporcion
# al tiempo t
tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
seed = int(input("Establece la semilla: "))
sim_polya = ECM.urna_polya(tiempos, seed)
ECM.graficar(sim_polya, "Simulación Urna de Polya", Polya = True)
