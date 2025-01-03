#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cadenas de Markov


Script para analizar cadenas de Markov, calcular distribuciones invariantes,
mixing times y realizar simulaciones.

Para utilizarlo, sigue las instrucciones 
y proporciona la información solicitada.


"""
import numpy as np
import funciones_CM as FCM

def inicializar_cadena_markov():
    """
    Inicializar la cadena de Markov obteniendo la matriz de transición.
    """
    longitud = FCM.longitud_espacio_estados()
    #El usuario se encargará de ingresar las probabilidades de transición
    #de la cadena para crear la matriz de transición
    P = FCM.matriz_transicion_(longitud)
    print("\nMatriz de Transición:\n" f"{P}\n")
    return P

def visualizar_cadena_markov(P):
    """
    Visualización de la cadena de Markov.
    """
    FCM.visualizar(P)

def clasificacion_estados(P):
    """
    Clasificación de los estados y análisis de las propiedades de la cadena
    de Markov.
    """

    #Empezaremos mostrando si tiene o no estados absorbentes
    est_abs = FCM.estados_absorbentes(P)
    if len(est_abs) == 0:
        print("La cadena no contiene estados absorbentes\n")
    else:
        print(f"Los estados absorbentes son {est_abs}\n")

    #Se muestran las diferentes clases de comunicación
    print("La cadena de Markov se compone de las siguientes clases:\n")
    FCM.tipo_clase(P)

    #Continuaremos diciendo si la cadena es irreducible
    clases_com = FCM.clases(P)
    if FCM.irreducible(clases_com):
        print("La cadena es irreducible\n")
    else:
        print("La cadena no es irreducible\n")

    #Calculo los periodos
    periodos = FCM.periodicidad(P)
    if FCM.irreducible(clases_com):
        print("Como la cadena es irreducible, todos los estados tienen \n"
              f"el mismo periodo y la cadena es de periodo {periodos[0]}.")
    else:
        print("Periodicidad por estado:\n")
        print(periodos)

    if FCM.aperiodica(P):
        print("Como la cadena es de periodo 1, entonces \nla cadena es aperiódica.\n")
    elif len(set(periodos.values())) == 1:
        print(f"La cadena es de periodo {periodos[0]}.\n")

    return clases_com

def analisis_distribucion_markov(P, clases_com):
    """
    Análisis de la distribución invariante/probabilidades de transición
    y, si es el caso, tiempo de mezcla.
    """
    clase_cerrada = FCM.c_cerradas(P)

    
    #Si la cadena es adecuada, es decir, con una única distribución 
    #invariante y aperiódica, calculo el mixing time y grafico las distancias
    #en cada tiempo hasta llegar a él.
    
    ##################################
    ##Caso 1: Irreducible y Aperiódica
    #################################

    if FCM.irreducible(clases_com) and FCM.aperiodica(P):
        dist_inv = FCM.dist_invariante(P)
        print("La cadena es irreducible y aperiódica. La distribución invariante pi es:\n"
              f"pi = {dist_inv}\n")

        error = float(input("Ingrese el error entre P y pi para determinar el tiempo de mezcla: "))
        t_mix = FCM.mixing_time(P, dist_inv, error)
        print(f"Mixing time t({error}) = {t_mix[0]}\n"
              f"Distancias en cada tiempo: {t_mix[1]}\n")
        FCM.grafica_TV(t_mix[1])

    ##################################
    ##Caso 2: Reducible y Aperiódica
    #################################

    elif not FCM.irreducible(clases_com) and FCM.aperiodica(P):
        print("La cadena no es irreducible pero es aperiódica. Se calculan probabilidades de absorción.")
        FCM.referenciar_absorcion(P)
        for A in clase_cerrada:
            P_abs, pA = FCM.matrices_abs(P, A)
            probabilidad_abs = np.round(FCM.prob_abs(P_abs, pA), decimals=5)
            print(f"Probabilidades de absorción para clase {A}: {probabilidad_abs}\n")

    ##################################
    ##Caso 3: Irreducible y Periódica
    #################################
    
    elif not FCM.aperiodica(P) and FCM.irreducible(clases_com):
        dist_inv = np.round(FCM.dist_invariante(P), decimals=5)
        print("La cadena es irreducible y periódica. Distribución invariante pi:\n"
              f"pi = {dist_inv}\n")

    ##################################
    ##Caso 4: Reducible y Periódica
    #################################
    
    elif not FCM.aperiodica(P) and not FCM.irreducible(clases_com):
        print("La cadena es reducible y periódica. Se calculan probabilidades de absorción.")
        #Indico las entradas de mi o mis vectores solución
        FCM.referenciar_absorcion(P)
        #Calculo todas las probabilidades de absorción para cada clase 
        #cerrada
        for A in clase_cerrada:
            P_abs, pA = FCM.matrices_abs(P, A)
            probabilidad_abs = np.round(FCM.prob_abs(P_abs, pA), decimals=5)
            print(f"Probabilidades de absorción para clase {A}: {probabilidad_abs}\n")

def simulacion_cadena_markov(P):
    """
    Simulación de la cadena de Markov dependiendo de la cantidad de tiempos
    que el usuario ingrese.
    """
    #Primero indicaremos su distribución inicial
    X0 = input("¿Desea que la distribución inicial sea uniforme? (Si/No): ")
    while X0 not in ["Si", "No"]:
        X0 = input("Por favor, ingrese 'Si' o 'No': ")

    if X0 == "Si":
        xo = np.random.choice(len(P), p=np.full(len(P), 1 / len(P)))
        print(xo)
    else:
        xo = np.zeros(len(P))
        for i in range(len(P)):
            while True:
                try:
                    prob = float(input(f"Ingrese la probabilidad inicial para el estado {i + 1}: "))
                    if 0 <= prob <= 1:
                        xo[i] = prob
                        break
                    else:
                        print("Por favor, ingrese un valor entre 0 y 1.")
                except ValueError:
                    print("Por favor, ingrese un número válido.")
        # Normalizar el vector xo para asegurar que la suma de las p
        #robabilidades sea 1
        xo /= np.sum(xo)

    tiempos = int(input("Ingrese la cantidad de tiempos para graficar: "))
    sim = FCM.simulacion(tiempos, P, xo)
    FCM.graficar(sim)


def main():
    Markov = True
    while Markov:
        P = inicializar_cadena_markov()
        visualizar_cadena_markov(P)
        clases_com = clasificacion_estados(P)
        analisis_distribucion_markov(P, clases_com)
        simulacion_cadena_markov(P)

        #Finalmente el usuario puede ingresar otra cadena para su análisis
        respuesta = input("¿Desea ingresar otra cadena de Markov? (Si/No): ")
        while respuesta not in ["Si", "No"]:
            respuesta = input("Por favor, ingrese 'Si' o 'No': ")
        if respuesta == "No":
            Markov = False

if __name__ == "__main__":
    main()
