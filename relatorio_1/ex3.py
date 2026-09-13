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
    primeira_iteracao = True

    if abs(f(xi)) < epsilon1: return xi, historico
    

    # usa a lib sympy pra criar uma expressão textual da derivada e converter ela pra função python
    symb_v =  sp.symbols('x')
    p, a, b, r, t = 10, 3.592, 0.04267, 0.08314, 300
    expr_fline = sp.diff((p + (a/symb_v**2)) * (symb_v - b) - (r*t), symb_v)
    f_line = sp.lambdify(symb_v, expr_fline, "numpy")
    
    while True:
        # chutamos um ponto x qualquer do plano cartesiano (xi). Pode estar longe ou próximo da raiz x
        # calculamos a derivada da função nesse ponto. Ou seja, a inclinação daquela curva no gráfico.
        # calculamos uma reta usando o ponto atual + derivada (taxa de inclinação) e vemos onde essa reta corta o eixo x.
        # esse x vira nosso próximo x, e ele está mais perto da raiz. Por que a derivada dita a inclinação.
        x = xi - (f(xi)/f_line(xi))

        if primeira_iteracao:
            erro_a = None
            primeira_iteracao = False
        else:
            erro_a = abs(x - xi) / abs(x) * 100
            erro_a = round(erro_a, 6)

        # aqui pode confundir, já que vi no enunciado é meu xi, e vi+1 é o meu x.
        historico.append([k, round(xi, 4), round(f(xi), 6), round(f_line(xi), 6), round(x, 4), erro_a])

        # se já estamos próximos o suficiente de 0 OU já convergiu (já ta próxima suficiente do valor real)
        # caso divirja, o meu newton raphson pode ficar rodando infinito.
        if abs(f(x)) < epsilon1 or abs(x - xi) < epsilon2:
            break

        # se não, move pro x mais próximo da raiz e tenta de novo.
        xi = x
        k = k + 1

    # print(k)
    return x, historico

# corta o eixo x em ~2.38
def f(v):
    t = 300
    p = 10
    a = 3.592
    b = 0.04267
    r = 0.08314

    return  (p + (a/v**2)) * (v - b) - (r*t)

# main

# fisicamente o comportamento real do Co2 não se desvia do comportamento ideal ou seja
# a fórmula dos gases ideais (rt/p) já é uma estimativa próxima da raiz, o que favorece a convergência rápida
# e estável do newton-raphson (q depende do chute inicial estar numa vizinhança próxima da solução)
v_ideal = (0.08314*300)/10
resultado, tabela_dados = newtonraphson(f, v_ideal, 0.001, 0.001)

# resposta da letra e
# O desvio de aproximadamente 4,21% entre o volume real (calculado pelo Newton-Raphson) 
# e o volume ideal confirma que a hipótese de gás ideal, embora seja uma simplificação, ainda é uma aproximação razoável nas condições do problema
# Ou seja o desvio entre comportamento ideal e real do gás é perceptível mas n é extremo

dif_perc = abs(resultado - v_ideal) / v_ideal * 100
print(dif_perc) # ~4.21

xs = np.linspace(0.5, 4, 200)
ys = f(xs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(xs, ys, label="f(v) = (p + (a/v**2)) * (v - b) - (r*t)")
ax1.spines["left"].set_position(("data", 2.25))
ax1.spines["bottom"].set_position("zero")
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.legend()

ax2.axis('off')
tabela = ax2.table(
    cellText = tabela_dados,
    colLabels = ["i", "vi", "f(vi)", "f'(vi)", "vi+1", "εa (%)"],
    loc = 'center'
)

plt.tight_layout()
plt.show()

