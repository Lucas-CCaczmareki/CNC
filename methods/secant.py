# o método de newton raphson faz parte dos métodos de ponto fixo
# o objetivo é o mesmo, aproximar o x que zere a f(x)

# método da secante - ruggiero
# ao invés de calclar a derivada igual ao newton-raphson, aproximamos ela
# utilizando 2 pontos já conhecidos da função

# os pontos iniciais não precisam cercar a raiz. São dois pontos quaisquer da função
# x0: ponto inicial 1
# x1: ponto inicial 2
# x2: novo ponto candidato, que está mais próximo da raiz

# epsilon1: quão próximo de zero o valor da função já tá. |f(x)|
# epsilon2: distância entre a nova aproximação e a anterior. |x - xi|

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    # função do slide, nosso objetivo é achar o x que faz o return ser = 0
    # ela zera próxima de 1.
    return x - 2*np.cos(x)

def secant(x0, x1, epsilon1, epsilon2):

    # se o primeiro ponto já tá próximo o suficiente de 0, retorna ele
    if abs(f(x0)) < epsilon1: return x0

    # se o segundo ponto já tá próximo suficiente de 0, retorna ele
    if abs(f(x1)) < epsilon1: return x1

    # se os 2 pontos iniciais já estão próximos os suficiente entre si a função convergiu.
    # ou seja, ela n vai chegar mais perto da raiz doq ela já tá. Então retorna x1 que é mais próximo da raiz
    if abs(x1 - x0) < epsilon2: return x1

    # inicia as iterações
    k = 1

    while True:
        # bloco1: f(x1) - f(x0). variação em y entre os 2 últimos pontos
        # bloco2: x1 - x0. variação em x entre os 2 últimos pontos
        # inclinação aproximada = bloco1 / bloco2
        # f(x1) / inclinação = quanto preciso andar em x pra reta secante zerar
        # (matematicamente igual a: (f(x1) / bloco1) * bloco2).
        # subtrair isso de x1 nos move na direção certa (esquerda ou direita) rumo à raiz
        x2 = x1 - (f(x1) / (f(x1) - f(x0))) * (x1 - x0)

        # se o x2(candidato) já tá próximo de zero o suficiente OU
        # se o intervalo entre o último ponto e o próximo já é pequeno suficiente (convergiu)
        # ent retorna x2 q ele já tá próximo suficiente da raiz.
        if abs(f(x2)) < epsilon1 or abs(x2 - x1) < epsilon2: 
            print(k)
            return x2

        # se não, move o intervalo dos pontos pra frente
        x0 = x1
        x1 = x2
        k = k + 1 # vai pra + 1 iteração

# main
result = secant(1, 6, 0.001, 0.001)

print(result)