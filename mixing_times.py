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

def main():
    
    Markov = True
    
    while Markov:
            
        """
        Definir la Cadena de Markov
        """
        #El usuario se encargará de ingresar las probabilidades de transición
        #de su cadena para crear la matriz de transición
        longitud = FCM.longitud_espacio_estados()
        P = FCM.matriz_transicion_(longitud)
        
         
       
        
        print("\nMatriz de Transición:\n"
              f"{P}\n")
    
        
        """
        Visualización de la cadena de Markov
        """
        FCM.visualizar(P)
        
        """
        Clasificación del espacio de estados
        """
        
        #Empezaremos mostrando si tiene o no estados absorbentes
        est_abs = FCM.estados_absorbentes(P)
        
    
        if len(est_abs) == 0:
            print("La cadena no contiene estados absorbentes\n")
        else:
            print(f"Los estados absorbentes son {est_abs}\n")
        
        #Muestro las diferentes clases de comunicación
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
            print("Como la cadena es de periodo 1, entonces \n"
                  "la cadena es aperiódica.\n")
        elif len(set(periodos.values())) == 1:
            print(f"La cadena es de periodo {periodos[0]}.\n")
        
        """
        Distribución invariante/
        Probabilidades de absorción
        """
        
        
        #Dependiendo la condición de mi cadena
        #muestro la o las distribuciones invariantes.
         
        clase_cerrada = FCM.c_cerradas(P)
       
        
        
        
        #Si la cadena es adecuada, es decir, con una única distribución 
        #invariante y aperiódica, calculo el mixing time y grafico las distancias
        #en cada tiempo hasta llegar a él.
        
        ##################################
        ##Caso 1: Irreducible y Aperiódica
        #################################
        
        if FCM.irreducible(clases_com) and FCM.aperiodica(P):
            dist_inv = FCM.dist_invariante(P)
            print(
                "La cadena al contar con una única distribución invariante\n"
                  " y ser aperiodica, su distribución invariante pi es la misma que"
                  "la límite.\n "
                  f"pi = {dist_inv} \n"
                  "Además, podemos hablar de la distancia que tiene\n"
                  "P respecto a pi en cada tiempo. \n"
                  )

            eigvals, eigvecs = np.linalg.eig(P)
            if (FCM.is_value_in_array(eigvals, 1)):
                error = float(input(
                    "Ingrese el error\n"
                                    " entre P y pi para determinar\n"
                                    "el tiempo de mezcla (mixing time): \n"
                                    ))
                t_mix = FCM.mixing_time(P, dist_inv, error)
                print(f"Mixing time t({error}) = {t_mix[0]}\n"
                      f"pues la distancia en cada tiempo fue {t_mix[1]}\n")
            else:
                error = float(input(
                    "Al contar con pérdida de decimales,\n"
                    "ingrese un error grande, para no tener problemas con la convergencia, \n"
                                    " entre P y pi para determinar\n"
                                    "el tiempo de mezcla (mixing time): \n"
                                    ))
                t_mix = FCM.mixing_time(P, dist_inv, error)
                print(f"Mixing time t({error}) = {t_mix[0]}\n"
                      f"pues la distancia en cada tiempo fue {t_mix[1]}\n")
           
            FCM.grafica_TV(t_mix[1])
        

        
        ##################################
        ##Caso 2: Reducible y Aperiódica
        #################################
        

        elif (not FCM.irreducible(clases_com) and FCM.aperiodica(P)):
            print(
                "Al no ser irreducible la cadena, pero sí aperiódica,\n"
                "se puede hablar de probabilidades de absorción para "
                "los estados transitorios\n"
            )
            
            # Referencia para absorción
            FCM.referenciar_absorcion(P)
        
            # Calcula todas las probabilidades de absorción para cada clase cerrada
            for A in clase_cerrada:
                P_abs, pA = FCM.matrices_abs(P, A)
                probabilidad_abs = np.round(FCM.prob_abs(P_abs, pA), decimals=5)
                
                print(
                    f"Las probabilidades de absorción respecto a la clase {A}\n"
                    f"están dadas por\n{probabilidad_abs}\n"
                )
                
                print("Además, podemos hablar de las distribuciones invariantes por cada clase cerrada:")
                
                for clase in clase_cerrada:
                    pi = np.round(FCM.dist_invariantes(P, clase), decimals=5)
                    print(f"{pi}\n") 
         
        ##################################
        ##Caso 3: Irreducible y Periódica
        #################################
        
        elif not FCM.aperiodica(P) and FCM.irreducible(P):
            
            dist_inv = np.round(FCM.dist_invariante(P), decimals=5)   
            print("Al ser periódica la cadena e irreducible\n"
                  "se puede hablar de la distribución invariante pi \n"
                  "(ésta no sería igual a la límite):\n"
                  f"pi = {dist_inv}")

        ##################################
        ##Caso 4: Reducible y Periódica
        #################################
            
        elif not FCM.aperiodica(P) and not FCM.irreducible(P):
            
            print("Al ser reducible y periódica la cadena,\n"
                  "se puede hablar de probabilidades de absorción para "
                  " los estados \n"
                  "transitorios\n")
            
            #Indico las entradas de mi o mis vectores solución
            FCM.referenciar_absorcion(P)
            
            #Calculo todas las probabilidades de absorción para cada clase 
            #cerrada
            for A in clase_cerrada:
                P_abs,pA = FCM.matrices_abs(P, A) 
                probabilidad_abs = np.round(FCM.prob_abs(P_abs, pA),decimals=5)
                
                print(f"Las probabilidades de absorción respecto a la clase "
                      "{A}\n"
                      "están dadas por\n"
                      f"{probabilidad_abs}\n")
                
                print("Además, podemos hablar de las distribuciones invariantes\n"
                      "por cada clase cerrada:")
                for clase in clase_cerrada:
                   pi = np.round(FCM.dist_invariantes(P, clase), decimals= 5)
                   
                   print(f"{pi}\n")
        
        """
        Simulación
        """
        
        #Aquí simularemos la cadena de Markov dependiendo de la cantidad de
        #tiempos que el usuario desee
    
        #Primero indicaremos su distribución inicial
        estado_inicial = True
        while estado_inicial:
            
            X0 = input("¿Desea que la distribución inicial sea uniforme?: Si, "
                       "No \n")
            
            while X0 not in ["Si", "No"]:
                X0 = input("Por favor, ingrese 'Si' o 'No': ")
            
            estado_inicial = False
            
        if X0 == "Si":
            
            xo = np.random.choice(len(P), p = np.full(len(P), 1/len(P)) )
            print(xo)    
            
            tiempos = int(input("Ingrese la cantidad de tiempos para "
                                "graficar: "))
            
            sim = FCM.simulacion(tiempos, P, xo)
            
            
            FCM.graficar(sim)
        else:
            
            for i in range(len(P)):
               
                while True:
                
                    try:
                        prob = float(input("Ingrese la probabilidad inicial "
                                           "para "
                                           f"el estado {i+1}: "))
                        if 0 <= prob <= 1:
                            xo[i] = prob
                            break
                    
                        else:
                            print("Por favor, ingrese un valor entre 0 y 1.")
                    
                    except ValueError:
                        print("Por favor, ingrese un número válido.")
            
            # Normalizar el vector xo para asegurar que la suma de las p
            #robabilidades sea 1
            xo = xo / np.sum(xo)
                
            print(xo)    
            
            tiempos = int(input("Ingrese la cantidad de tiempos para "
                                "graficar: "))
            
            sim = FCM.simulacion(tiempos, P, xo)
            
        #Finalmente el usuario puede ingresar otra cadena para su análisis
        respuesta = input("¿Desea ingresar otra cadena de Markov?: Si, No \n")
        
        while respuesta not in ["Si", "No"]:
            respuesta = input("Por favor, ingrese 'Si' o 'No': ")
        
        if respuesta == "No":
            Markov = False
            

if __name__ == "__main__":
    main()
