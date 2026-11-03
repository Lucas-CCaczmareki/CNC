# o método de newton raphson faz parte dos métodos de ponto fixo
# o objetivo é o mesmo, aproximar o x que zere a f(x)

# xi: aproximação inicial
# x: novo candidato pra aproximar
# epsilon1: quão próximo de zero o valor da função já tá. |f(x)|
# epsilon2: distância entre a nova aproximação e a anterior. |x - xi|

# ps: epsilon2 indica o quão próxima a função está de convergir. Se ele n mudou mt significa que ela já convergiu

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    # função do slide, nosso objetivo é achar o x que faz o return ser = 0
    return x - 2*np.cos(x)

def newtonraphson(xi, epsilon1, epsilon2):

    if abs(f(xi)) < epsilon1: return xi 
    k = 1

    # usa a lib sympy pra criar uma expressão textual da derivada e converter ela pra função python
    symb_xi =  sp.symbols('x')
    expr_fline = sp.diff((symb_xi - 2*sp.cos(symb_xi)), symb_xi)
    f_line = sp.lambdify(symb_xi, expr_fline, "numpy")
    
    while True:
        # chutamos um ponto x qualquer do plano cartesiano (xi). Pode estar longe ou próximo da raiz x
        # calculamos a derivada da função nesse ponto. Ou seja, a inclinação daquela curva no gráfico.
        # calculamos uma reta usando o ponto atual + derivada (taxa de inclinação) e vemos onde essa reta corta o eixo x.
        # esse x vira nosso próximo x, e ele está mais perto da raiz. Por que a derivada dita a inclinação.
        x = xi - (f(xi)/f_line(xi))

        # se já estamos próximos o suficiente de 0 OU já convergiu (já ta próxima suficiente do valor real)
        # caso divirja, o meu newton raphson pode ficar rodando infinito.
        if abs(f(x)) < epsilon1 or abs(x - xi) < epsilon2:
            break

        # se não, move pro x mais próximo da raiz e tenta de novo.
        xi = x
        k = k + 1

    print(k)
    return x

# main
result = newtonraphson(1, 0.001, 0.001)
print(result)
    
