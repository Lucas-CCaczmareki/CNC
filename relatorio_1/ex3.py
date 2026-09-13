import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# xi: aproximação inicial
# x: novo candidato pra aproximar
# epsilon1: quão próximo de zero o valor da função já tá. |f(x)|
# epsilon2: distância entre a nova aproximação e a anterior. |x - xi|
# ps: epsilon2 indica o quão próxima a função está de convergir. Se ele n mudou mt significa que ela já convergiu

def newtonraphson(f, xi, epsilon1, epsilon2):
    k = 1
    historico = []
    x_anterior = None

    if abs(f(xi)) < epsilon1: return xi, historico
    

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

        if x_anterior is None:
            erro_a = None
        else:
            erro_a = abs(x - x_anterior) / abs(x) * 100
            erro_a = round(erro_a, 6)

        historico.append([k, round(x, 4), round(f(x), 6), round(f_line(x), 6), round(xi, 4), erro_a])

        # se já estamos próximos o suficiente de 0 OU já convergiu (já ta próxima suficiente do valor real)
        # caso divirja, o meu newton raphson pode ficar rodando infinito.
        if abs(f(x)) < epsilon1 or abs(x - xi) < epsilon2:
            break

        # se não, move pro x mais próximo da raiz e tenta de novo.
        xi = x
        k = k + 1

    # print(k)
    return x, historico



