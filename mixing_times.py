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
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def longitud_espacio_estados():
    """
    Esta función se encarga de preguntar al usuario sobre el número de estados
    en su cadena de Markov. Se espera que reciba un número no negativo por
    parte del usuario.

    Returns
    -------
    longitud : int
        La longitud del espacio de estados n.

    """
    while True:  
        try:
            longitud = int(input("Ingrese la longitud del "
                                 "espacio de estados: ")) 
            
            if longitud > 0: 
                return longitud  
            
            else:
                print("Ingrese un valor positivo mayor que cero.")
                
        except ValueError:
            print("Ingrese un número entero válido.") 


def matriz_transicion_(longitud):
    """
    Esta función crea la matriz de transición de la Cadena de Markov. 
    Se recomienda que si es una cadena irreducible o con una clase recurrente,
    sea aperiódica.

    Parameters
    ----------
    longitud : int
        Número que indica las columnas y renglones a crear en la matriz.

    Returns
    -------
    M = np.array
        Matriz con las probabilidades de transición entre cada estado de 
        la cadena.

    """
    matriz_transicion = []
    
    #Recorro cada entrada de la matriz para que el usuario indique 
    #la probabilidad de transición
    for i in range(longitud):
        fila = []
        prob_suma = 0
        
        for j in range(longitud):
         
            while True:
                try:
                    prob = float(input(f"Ingrese la probabilidad de transición"
                                       f" de {i+1} a {j+1}: "))
                    if 0 <= prob <= 1:
                        fila.append(prob)
                        prob_suma += prob
                        break
                    
                    else:
                        print("Por favor, ingrese un valor entre 0 y 1.")
                
                except ValueError:
                    print("Por favor, ingrese un número válido.")
                    
        # Normalizar la fila para asegurar que la suma de las 
        # probabilidades sea 1
        fila = [p / prob_suma for p in fila]
        matriz_transicion.append(fila)
        M = np.array(matriz_transicion) 
    return M


def estados_absorbentes(matriz):
    """
    Esta función te brindará información si tu cadena contiene estados 
    absorbentes

    Parameters
    ----------
    matriz : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    estados : list
        Una lista que contiene a todos los estados absorbentes de la cadena.

    """
    #Checo si cada entrada [i,i] de la matriz = 1 para determinar
    #si es absorbente y añadirlos a la lista
    estados = []
    
    for i in range(len(matriz)):
        es_absorbente = True
        
        for j in range(len(matriz[i])):
            
            if j != i and matriz[i, j] != 0:
                es_absorbente = False
                break
            
        if es_absorbente:
            estados.append(i)
    return estados




def clases(P):
    """
    Esta función se encarga de indicarte las clases de comunicación de la
    cadena de Markov.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    classes : list
        Lista de conjuntos que representan a las diferentes clases de
        comunicación.

    """
    G = nx.DiGraph(P)

    # Encuentra las clases
    classes = list(nx.strongly_connected_components(G))

    return classes


def irreducible(clases):
    """
    Función que te indica si tu cadena es irreducible

    Parameters
    ----------
    clases : list
        Lista que contiene a las diferentes clases de comunicación.

    Returns
    -------
    bool
        Si tu lista tiene solo 1 elemento, es decir, solo 1 clase de 
        comunicación, te regresa True y por tanto, indica que es irreducible
        la cadena.

    """
    
    if len(clases)==1:
        return True
    else:
        return False



def submatriz(P, clase):
    """
    Función que te da la matriz de transición de una clase de comunicación
    específica.    

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.
    clase : set
        Conjunto de estados que pertenecen a la misma clase de comunicación.

    Returns
    -------
    submatriz : np.array
        Es una submatriz que solo contiene las probabilidades de transición
        de una clase.

    """
    
    nodos_clase = list(clase)
    
    #Te da la submatriz dependiendo tu clase
    submatriz = P[np.ix_(nodos_clase, nodos_clase)]
    
    return submatriz



def submatriz_clase(P):
    """
    Función que nos dará todas las submatrices de únicamente clases cerradas
    de la cadena.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    P_sub : list
        Lista que tiene como elementos np.array (matrices). La longitud de mi
        lista dependerá de las clases cerradas de la cadena.

    """    
    P_sub =[]
    
    #Ciclo que te añade en la lista las submatrices
    for clase in clases(P):
        P_c = submatriz(P, clase)
        #sumo si las filas de la submatriz dan cercano a 1
        #para así determinar si es cerrada la clase
        suma_filas = np.sum(P_c, axis=1)
        resultado = np.isclose(suma_filas, 1.0)
        
        if np.all(resultado):
            P_sub.append(P_c)
            
    return P_sub

    
def tipo_clase(P):
    """
    Función que imprime las clases de comunicación y te indica si son cerradas
    o abiertas.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    None.

    """
    
    for clase in clases(P):
        P_c = submatriz(P, clase)
        suma_filas = np.sum(P_c, axis=1)
        #Sumo las filas de mis submatrices y veo si son lo suficientemente
        #cercanas a 1 para determinar si es cerrada o abierta la clase
        resultado = np.isclose(suma_filas, 1.0)
        
        
        if np.all(resultado):
            print(f"La clase {clase} es:\n"
                  "Cerrada\n")
        else:
            print(f"La clase {clase} es:\n"
                  "Abierta\n")



def matrices_abs(P, A):
    """
    Función que me regresará dos importantes matrices: la submatriz de las 
    probabilidades de transición entre todos los estados transitorios y 
    el vector que indica la probabilidad de transición de cada estado 
    transitorio hacia la clase cerrada A.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.
    A : set
        Clase cerrada de mi cadena.

    Returns
    -------
    M : np.array
        Matriz que contiene únicamente las probabilidades de transición
        entre los estados transitorios.
    pA : np.array
        Vector que contiene las probabilidades de transición de los 
        estados transitorios hacia la clase cerrada A.

    """
    
    
    clase_abierta = c_abiertas(P)
    clases_cerradas = c_cerradas(P)
    
    
    #Mover la clase cerrada A al final 
    clases_cerradas = sorted(clases_cerradas, key=lambda x: 'A' not in x)
    
    estados_transitorios = []
    
    for clase in clase_abierta:
        for estados in clase:
            estados_transitorios.append(estados)
    
    nEst_tran = len(estados_transitorios)
    
    
    M = np.empty((nEst_tran, nEst_tran))
    pA = np.empty((nEst_tran, 1))
    
    #1. Integro primero las probabilidades que conocemos entre estados
    # transitorios
    for i,l in enumerate(estados_transitorios):
        for j,k in enumerate(estados_transitorios):
            M[i,j] = P[l,k]
            
    #2. Integro en pA las probabilidades de todos los estados a A    
        for i,l in enumerate(estados_transitorios):
            pA[i,0] = np.sum(P[l,list(A)])
    
    return M, pA



def referenciar_absorcion(P):
    """
    Función que, al momento de querer calcular las probabilidades de absorción
    de los estados transitorios, me indique a qué estado corresponde cada
    entrada de mi vector solución de probabilidades de absorción.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    None.

    """
    
    
    clase_abierta = c_abiertas(P)
    
    
    
    #Obtengo los estados transitorios, pues son aquellos a los que se les 
    #calculará la proba de absorción
    estados_transitorios = []
    
    for clase in clase_abierta:
        for estados in clase:
            estados_transitorios.append(estados)
    
   
    
    #Referencio las columnas y renglones para su lectura
    for i,l in enumerate(estados_transitorios):
        print(f"Mi entrada {i} refiere a la probabilidad\n"
              f"de absorción de mi estado {l}\n")



def prob_abs(P, pA):  
    """
    Función que me calculará las probabilidades de absorción de mis estados 
    transitorios hacia la clase cerrada A.
    El sistema a resolver es 
    [P_{A1UA2U..}^c - Id]h = -pA
    Donde A1,A2,... son mis clases cerradas y
    P_{A1UA2U...}^c es mi matriz restringida a todos mis estados transitorios
    
    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición entre mis estados transitorios.
    pA : np.array
        Probabilidad de los estados transitorios hacia la clase cerrada A.

    Returns
    -------
    h : np.array
        Vector solución que indica las probabilidades de transición de 
        cada estado transitorio i hacia la clase cerrada A.

    """
    A = P - np.eye(len(P))
    
    # Construir el lado derecho del sistema
    b = -pA
    
    # Resolver el sistema de ecuaciones lineales
    # [P_{A1UA2U..}^c - Id]h = -pA
    h = np.linalg.solve(A, b)
    h = np.round(h, 3)
    
    return h



def c_cerradas(P):
    """
    Función que me regresará una lista con las clases cerradas de mi cadena

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    clases_cl : list
        Lista que contiene únicamente las clases cerradas de la cadena 
        de Markov.

    """
    clases_cl = []
    
    cs = clases(P)
    
    #Por cada clase checo si cumple con la condición de que si
    #todas las filas de su submatriz son 1 para determinar si son cerradas
    for clase in cs:
        P_c = submatriz(P, clase)
        suma_filas = np.sum(P_c, axis=1)
        resultado = np.isclose(suma_filas, 1.0)
        if np.all(resultado):
            clases_cl.append(clase)
    return clases_cl    
    


def c_abiertas(P):
    """
    Función que me regresará una lista con las clases abiertas de mi cadena

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    clases_cl : list
        Lista que contiene únicamente las clases abiertas de la cadena 
        de Markov.

    """
    clases_op = []
    
    cs = clases(P)
    
    #Por cada clase checo si cumple con la condición de que si
    #todas las filas de su submatriz no son 1 para determinar si son abieras
    for clase in cs:
        P_c = submatriz(P, clase)
        suma_filas = np.sum(P_c, axis=1)
        resultado = np.isclose(suma_filas, 1.0)
        
        if not np.all(resultado):
            clases_op.append(clase)
            
    return clases_op    


def dist_invariante(P):
    """
    Función que te da la distribución estacionaria/invariante de tu cadena,
    si esta lo permite.
    Ecuación a resolver
    (P^t - Id)X = 0

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    solucion : np.array
        Lista o vector que representa la distribución invariante.

    """
    
    P_traspuesta = np.transpose(P)
    
    A = P_traspuesta - np.identity(len(P))
    # Hacer mi última fila de 1's por cuestiones de dependencia lineal
    A[-1] = 1
    
    b = np.zeros(len(P))
    # Hacer mi última entrada igual a 1 por cuestiones de dependencia lineal
    b[-1] = 1
    

    # Resolver el sistema de ecuaciones
    # AX = 0 o (P^t - Id)X = 0
    solucion = np.linalg.solve(A, b)
    solucion = np.round(solucion, 7) 
    return solucion


def dist_invariantes(P, clase_cl):
    """
    Función que te dará la distribución invariante respecto
    a alguna clase cerrada.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.
    clase_cl : set
        Clase cerrada a la que nos interesa sacar su distribución invariante.

    Returns
    -------
    dist_inv_extendida : np.array
        Es una vector que tiene la distribución invariante para una clase 
        cerrada; este vector se caracteriza porque tiene entradas 0 en los
        estados fuera de la clase cerrada.

    """
    nodos_clase = list(clase_cl)

    P_c = submatriz(P, clase_cl)
    
    P_traspuesta = np.transpose(P_c)
    
    A = P_traspuesta - np.identity(len(nodos_clase))
    A[-1] = 1
    
    b = np.zeros(len(nodos_clase))
    b[-1] = 1

    # Resolver el sistema de ecuaciones
    solucion = np.linalg.solve(A, b)
    solucion = np.round(solucion, 7)
    
    # Crear el vector de distribución invariante extendido a la cadena completa
    # es decir, le añado 0's a las entadas de los estados
    # que no son de la clase.
    dist_inv_extendida = np.zeros(len(P))
    dist_inv_extendida[nodos_clase] = solucion
    return dist_inv_extendida


def periodicidad(P):
    """
    Función que te permite conocer los periodos para cada estado

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    periodos : dict
        Diccionario que indica el estado y su periodo correspondiente.

    """
    G = nx.DiGraph(P)

    # Crear un diccionario para los periodos
    periodos = {}
    
    for estado in G.nodes():
        ciclos = nx.simple_cycles(G.subgraph(list(G.predecessors(estado)) + [estado]))
        longitudes_ciclos = [len(ciclo) for ciclo in ciclos]
        
        # Verificar si se encontraron ciclos para el estado actual
        if longitudes_ciclos:
            periodo = np.gcd.reduce(longitudes_ciclos)
            periodos[estado] = periodo
        else:
            periodos[estado] = None
    
    return periodos


def aperiodica(P):
    """
    Función que permite conocer si la cadena de Markov es aperiódica.

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición..

    Returns
    -------
    ap : Boolean
        Te indica si la cadena es aperiódica.

    """
    periodos = list(periodicidad(P).values())
    ap = all(d == 1 for d in periodos)
    return ap

    

def distancia_max_TV(matriz_trans, dinv, t):
    """
    Función que dará la métrica que se utiliza para medir la distancia
    de tu cadena de Markov hacia la distribución invariante. La cadena debe de
    tener una única distribución invariante.
    d(t) = max_i{(1/2) sum_{j in S} | P_{ij} - pi_{j}| }
    

    Parameters
    ----------
    matriz_trans : np.array
        Matriz de probabilidades de transición.
    dinv : np.array
        Distribución invariante de tu cadena.
    t : int
        Número que indica el tiempo en el que te encuentras dentro de
        tu cadena.

    Returns
    -------
    dist : float
        Número que indica la distancia a la que te encuentras de la 
        distribución invariante.

    """
    
    ##Ojo vamos a diagonalizar la matriz
    
    # Calcular los valores propios y los vectores propios
    eigvals, eigvecs = np.linalg.eig(matriz_trans)
    
    # Crear la matriz diagonal D a partir de los valores propios
    D = np.diag(eigvals)
    
    # La matriz P es simplemente los vectores propios
    C = eigvecs
    
    # Calcular P inversa
    C_inv = np.linalg.inv(C)
    
    
    # Resolvemos 
    D_t = np.linalg.matrix_power(D, t)
    
    P_t = C @ D_t @ C_inv
    
    
    n = len(P_t)
    M = np.zeros((n, n))
    TV = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            M[i, j] = np.abs(P_t[i, j] - dinv[j])
        TV[i] = 0.5 * np.sum(M[i, :])
        
    


    dist = np.max(TV)
    
    return dist


def mixing_time(P, dist_inv, error):
    """
    Te indica a partir de qué tiempo/paso/momento estás menor a una distancia
    dada, de tu distribución invariante.
    
    t(error) = min{t: d(t) <= error}
    
    Así se define el mixing time
    

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.
    dist_inv : np.array
        Vector de la distribución invariante.
    error : float
        Número flotante pequeño (entre 0 y 1) del cual quieres saber 
        hasta que tiempo alcanzas una distancia menor a éste.

    Returns
    -------
    t : int
        Tiempo en el que se encuentra la distancia debajo del error
    distancia_tiempo : list
        Lista que contiene la distancia hacia la distribución invariante
        en cada tiempo, con una longitud de "t(error)".
    

    """
    
    #Almacenará la distancia de cada tiempo
    distancia_tiempo = [] 
    t = 1
    
    while True:
        distancia_tiempo.append(distancia_max_TV(P, dist_inv, t))
        #Elevo la matriz a la siguiente potencia si no logro ser menor que 
        #el error dado
        if distancia_tiempo[t-1] > error:
            t += 1
        else:
            break

    return t, distancia_tiempo




def grafica_TV(distancias):
    """
    Función que graficará la distancia hacia la distribución invariante en 
    cada tiempo.

    Parameters
    ----------
    distancias : list
        Lista que en cada entrada contiene la distancia en cada tiempo hacia 
        la distribución invariante.

    Returns
    -------
    None.

    """

    df = pd.DataFrame({
    "d(t)": distancias,  
    "tiempo": range(1, len(distancias) + 1)
    })
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(df["tiempo"], df["d(t)"], color="black")
    plt.plot(df["tiempo"], df["d(t)"],  color="red")
    plt.xlabel("Tiempo")
    plt.ylabel("d(t)")
    plt.title("Distancia en Cada Tiempo Respecto a la Distribución "
              "Estacionaria")
    plt.legend()
    plt.show()



def simulacion(pasos, matriz, xo):
    """
    Función que te simulará una trayectoria dentro de la cadena de Markov

    Parameters
    ----------
    pasos : int
        Número de pasos/tiempos a simular.
    matriz : np.array
        Matriz de probabilidades de transición de la cadena.
    xo : int
        Es el estado inicial de la cadena.

    Returns
    -------
    lista : list
        Lista que contiene la trayectoria de la cadena. Es tan larga tantos 
        pasos se hayan indicado.

    """
    
    lista = list()
    lista.append(xo)
    
    #Simulo la trayectoria checando las probabilidades
    # de mi matriz. 
    for i in range(1,pasos):
        lista.append((np.random.choice(range(0,len(matriz)), 
                                       p = matriz[lista[i-1],])))
    
    return lista


def graficar(simulacion):
    """
    Función que graficará la simulación de la cadena de Markov.

    Parameters
    ----------
    simulacion : list
        Lista que indica el estado en cada tiempo de la simulación 
        de tu cadena.

    Returns
    -------
    None.

    """
    
    df = pd.DataFrame({
        "estados": simulacion,  
        "tiempo": range(1, len(simulacion) + 1)
    })
    
    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(df["tiempo"], df["estados"], color="black")
    plt.plot(df["tiempo"], df["estados"], color="blue")
    plt.xlabel("Tiempo")
    plt.ylabel("Estado")
    plt.title("Simulacion Cadena de Markov")
    plt.xticks(df["tiempo"])  # Coloca marcas en el eje x en todos los valores de tiempo
    plt.yticks(range(int(min(simulacion)), int(max(simulacion)) + 1))  # Coloca marcas en el eje y en los valores enteros
    plt.legend()
    plt.grid(True)  # Agrega una cuadrícula al gráfico
    plt.show()


def visualizar(P):
    """
    Función que permite visualizar la cadena de Markov como una gráfica 
    dirigida

    Parameters
    ----------
    P : np.array
        Matriz de probabilidades de transición.

    Returns
    -------
    None.

    """
    
    G = nx.DiGraph(P)
    
    # Agregar nodos al grafo
    for i in range(len(P)):
        G.add_node(i)
    
    # Agregar aristas ponderadas al grafo
    for i in range(len(P)):
        for j in range(len(P)):
            if P[i][j] > 0:
                G.add_edge(i, j, weight=P[i][j])
    
    # Dibujar el grafo
    
    # Posiciones de los nodos
    pos = nx.circular_layout(G)  
    
    # Etiquetas de los nodos
    labels = {i: f"{i}" for i in range(len(P))} 
    
    # Etiquetas de las aristas
    edge_labels = {(i, j): round(P[i][j], 2) for i in range(len(P))
                   for j in range(len(P)) if P[i][j] > 0} 
    
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, 
            node_color='skyblue', font_size=12, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('Grafo de la Cadena de Markov')
    plt.show()
    
def is_value_in_array(array, value):
    """
    Función que permitirá determinar si dentro de un array se encuentra un cierto
    valor, con base en qué tanto se parecen dados sus decimales.

    Parameters
    ----------
    array : np.array
        Es el array al que eximinará.
    value : float
        Es el valor deseado para buscar dentro del array.

    Returns
    -------
    bool
        Te indica si se encuentra el valor dado dentro del array.

    """
    for elem in array:
        if np.round(elem, decimals=10) == value:
            return True
    return False

def main():
    
    Markov = True
    
    while Markov:
            
        """
        Definir la Cadena de Markov
        """
        #El usuario se encargará de ingresar las probabilidades de transición
        #de su cadena para crear la matriz de transición
        longitud = longitud_espacio_estados()
        P = matriz_transicion_(longitud)
        
         
       
        # Ésta funciona
        P = np.array([[0.75, 0.05, 0.2], 
                      [0.8, 0.1, 0.1],
                      [0.3, 0.2, 0.5]])
        
        P = np.array([[0, 0.3, 0.1, 0.6],
                     [0.4, 0, 0.6, 0],
                     [0.2, 0.2, 0, 0.6],
                     [0.5, 0.1, 0.4, 0]])
        P =np.array([
                [0, 0.2, 0.1, 0.2, 0.5],
                [0.3, 0, 0.2, 0.3, 0.2],
                [0.4, 0.1, 0, 0.4, 0.1],
                [0.1, 0.3, 0.3, 0, 0.3],
                [0.2, 0.4, 0.2, 0.2, 0]
            ])
        
        P = np.array([[0.2,0.1,0.05,0.05,0.3,0.3],
                      [0.05,0.025,0.05,0.025,0.05,0.8],
                      [0.25,0, 0.3, 0.25,0, 0.2],
                      [0.1, 0.3, 0.3,0 , 0.2, 0.1],
                      [0, 0, 0.33 , 0.33, 0, 0.34],
                      [0.4, 0.1, 0, 0.4, 0.1, 0]])
        
        print("\nMatriz de Transición:\n"
              f"{P}\n")
    
        
        """
        Visualización de la cadena de Markov
        """
        visualizar(P)
        
        """
        Clasificación del espacio de estados
        """
        
        #Empezaremos mostrando si tiene o no estados absorbentes
        est_abs = estados_absorbentes(P)
        
    
        if len(est_abs) == 0:
            print("La cadena no contiene estados absorbentes\n")
        else:
            print(f"Los estados absorbentes son {est_abs}\n")
        
        #Muestro las diferentes clases de comunicación
        print("La cadena de Markov se compone de las siguientes clases:\n")
        tipo_clase(P)
        
        #Continuaremos diciendo si la cadena es irreducible
        clases_com = clases(P)
        
        if irreducible(clases_com):
            print("La cadena es irreducible\n")
        else:
            print("La cadena no es irreducible\n")
        
        
        #Calculo los periodos
        periodos = periodicidad(P)
        if irreducible(clases_com):
            print("Como la cadena es irreducible, todos los estados tienen \n"
                  f"el mismo periodo y la cadena es de periodo {periodos[0]}.")
        else:
            print("Periodicidad por estado:\n")
            print(periodos)

        
        if aperiodica(P):
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
         
        clase_cerrada = c_cerradas(P)
       
        
        
        
        #Si la cadena es adecuada, es decir, con una única distribución 
        #invariante y aperiódica, calculo el mixing time y grafico las distancias
        #en cada tiempo hasta llegar a él.
        
        ##################################
        ##Caso 1: Irreducible y Aperiódica
        #################################
        #Establecer un error min para que con base en él redondee los decimales.
    
        #mixing_time(P, dist_inv, 0.005)
        #distancia_max_TV(P, dist_inv, 10)
        
        if irreducible(clases_com) and aperiodica(P):
            dist_inv = dist_invariante(P)
            print(
                "La cadena al contar con una única distribución invariante\n"
                  " y ser aperiodica, su distribución invariante pi es la misma que"
                  "la límite.\n "
                  f"pi = {dist_inv} \n"
                  "Además, podemos hablar de la distancia que tiene\n"
                  "P respecto a pi en cada tiempo. \n"
                  )

            eigvals, eigvecs = np.linalg.eig(P)
            if (is_value_in_array(eigvals, 1)):
                error = float(input(
                    "Ingrese el error\n"
                                    " entre P y pi para determinar\n"
                                    "el tiempo de mezcla (mixing time): \n"
                                    ))
                t_mix = mixing_time(P, dist_inv, error)
                print(f"Mixing time t({error}) = {t_mix[0]}\n"
                      f"pues la distancia en cada tiempo fue {t_mix[1]}\n")
            else:
                error = float(input(
                    "Al contar con pérdida de decimales,\n"
                    "ingrese un error grande, para no tener problemas con la convergencia, \n"
                                    " entre P y pi para determinar\n"
                                    "el tiempo de mezcla (mixing time): \n"
                                    ))
                t_mix = mixing_time(P, dist_inv, error)
                print(f"Mixing time t({error}) = {t_mix[0]}\n"
                      f"pues la distancia en cada tiempo fue {t_mix[1]}\n")
           
            grafica_TV(t_mix[1])
        

        
        ##################################
        ##Caso 2: Reducible y Aperiódica
        #################################
        

        elif (not irreducible(clases_com) and aperiodica(P)):
            print(
                "Al no ser irreducible la cadena, pero sí aperiódica,\n"
                "se puede hablar de probabilidades de absorción para "
                "los estados transitorios\n"
            )
            
            # Referencia para absorción
            referenciar_absorcion(P)
        
            # Calcula todas las probabilidades de absorción para cada clase cerrada
            for A in clase_cerrada:
                P_abs, pA = matrices_abs(P, A)
                probabilidad_abs = np.round(prob_abs(P_abs, pA), decimals=5)
                
                print(
                    f"Las probabilidades de absorción respecto a la clase {A}\n"
                    f"están dadas por\n{probabilidad_abs}\n"
                )
                
                print("Además, podemos hablar de las distribuciones invariantes por cada clase cerrada:")
                
                for clase in clase_cerrada:
                    pi = np.round(dist_invariantes(P, clase), decimals=5)
                    print(f"{pi}\n") 
         
        ##################################
        ##Caso 3: Irreducible y Periódica
        #################################
        
        elif not aperiodica(P) and irreducible(P):
            
            dist_inv = np.round(dist_invariante(P), decimals=5)   
            print("Al ser periódica la cadena e irreducible\n"
                  "se puede hablar de la distribución invariante pi \n"
                  "(ésta no sería igual a la límite):\n"
                  f"pi = {dist_inv}")

        ##################################
        ##Caso 4: Reducible y Periódica
        #################################
            
        elif not aperiodica(P) and not irreducible(P):
            
            print("Al ser reducible y periódica la cadena,\n"
                  "se puede hablar de probabilidades de absorción para "
                  " los estados \n"
                  "transitorios\n")
            
            #Indico las entradas de mi o mis vectores solución
            referenciar_absorcion(P)
            
            #Calculo todas las probabilidades de absorción para cada clase 
            #cerrada
            for A in clase_cerrada:
                P_abs,pA = matrices_abs(P, A) 
                probabilidad_abs = np.round(prob_abs(P_abs, pA),decimals=5)
                
                print(f"Las probabilidades de absorción respecto a la clase "
                      "{A}\n"
                      "están dadas por\n"
                      f"{probabilidad_abs}\n")
                
                print("Además, podemos hablar de las distribuciones invariantes\n"
                      "por cada clase cerrada:")
                for clase in clase_cerrada:
                   pi = np.round(dist_invariantes(P, clase), decimals= 5)
                   
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
            
            sim = simulacion(tiempos, P, xo)
            
            
            graficar(sim)
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
            
            sim = simulacion(tiempos, P, xo)
            
        #Finalmente el usuario puede ingresar otra cadena para su análisis
        respuesta = input("¿Desea ingresar otra cadena de Markov?: Si, No \n")
        
        while respuesta not in ["Si", "No"]:
            respuesta = input("Por favor, ingrese 'Si' o 'No': ")
        
        if respuesta == "No":
            Markov = False
            

if __name__ == "__main__":
    main()
