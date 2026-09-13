import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# os pontos iniciais não precisam cercar a raiz. São dois pontos quaisquer da função
# x0: ponto inicial 1
# x1: ponto inicial 2
# x2: novo ponto candidato, que está mais próximo da raiz

# epsilon1: quão próximo de zero o valor da função já tá. |f(x)|
# epsilon2: distância entre a nova aproximação e a anterior. |x - xi|
def secant(f, x0, x1, epsilon1, epsilon2):

    # inicia as iterações
    k = 1
    historico = []
    primeira_iteracao = True

    # se o primeiro ponto já tá próximo o suficiente de 0, retorna ele
    if abs(f(x0)) < epsilon1: return x0, historico

    # se o segundo ponto já tá próximo suficiente de 0, retorna ele
    if abs(f(x1)) < epsilon1: return x1, historico

    # se os 2 pontos iniciais já estão próximos os suficiente entre si a função convergiu.
    # ou seja, ela n vai chegar mais perto da raiz doq ela já tá. Então retorna x1 que é mais próximo da raiz
    if abs(x1 - x0) < epsilon2: return x1, historico

    while True:
        # bloco1: f(x1) - f(x0). variação em y entre os 2 últimos pontos
        # bloco2: x1 - x0. variação em x entre os 2 últimos pontos
        # inclinação aproximada = bloco1 / bloco2
        # f(x1) / inclinação = quanto preciso andar em x pra reta secante zerar
        # (matematicamente igual a: (f(x1) / bloco1) * bloco2).
        # subtrair isso de x1 nos move na direção certa (esquerda ou direita) rumo à raiz
        x2 = x1 - (f(x1) / (f(x1) - f(x0))) * (x1 - x0)

        if primeira_iteracao:
            erro_a = None
            primeira_iteracao = False
        else:
            erro_a = round(abs(x2 - x1) / abs(x2) * 100, 6)

        historico.append([k, round(x0, 4), round(x1, 4), round(f(x0), 6), round(f(x1), 6), round(x2, 4), erro_a])

        # se o x2(candidato) já tá próximo de zero o suficiente OU
        # se o intervalo entre o último ponto e o próximo já é pequeno suficiente (convergiu)
        # ent retorna x2 q ele já tá próximo suficiente da raiz.
        if abs(f(x2)) < epsilon1 or abs(x2 - x1) < epsilon2: 
            print(k)
            return x2, historico

        # se não, move o intervalo dos pontos pra frente
        x0 = x1
        x1 = x2
        k = k + 1 # vai pra + 1 iteração

def f(vd):
    vs = 5
    r = 1000
    sat_i = 1 * 10**(-12)
    n = 1.7
    vt = 25.85 * 10**(-3)

    return (sat_i * (np.exp(vd/(n*vt)) - 1)) - ((vs - vd)/r)

# main
vd_vals = np.linspace(0, 1.1, 300)

vs = 5
r = 1000
sat_i = 1 * 10**(-12)
n = 1.7
vt = 25.85 * 10**(-3)

Id_vals = sat_i * (np.exp(vd_vals/(n*vt)) - 1)
Ir_vals = (vs - vd_vals) / r

resultado, tabela_dados = secant(f, 0.95, 1.00, 0.001, 0.001)
print(resultado)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))

# LETRA B
# A interseção das curvas é o ponto de operação real do circuito
# onde a corrente que o diodo permite passar coincide com a que a fonte/resistor fornecem, 
# já que num circuito série só existe uma corrente.

# gráfico 1: I_D e I_R no mesmo sistema de coordenadas
ax1.plot(vd_vals, Id_vals, label="I_D(V_D)")
ax1.plot(vd_vals, Ir_vals, label="I_R(V_D)")
ax1.legend()

# gráfico 2: f(V_D)
ys = f(vd_vals)
ax2.plot(vd_vals, ys, label="f(V_D)")
ax2.axhline(0, color="gray")
ax2.legend()

# item d: tabela
ax3.axis('off')
ax3.table(
    cellText = tabela_dados,
    colLabels =["i", "V_(i-1)", "V_i", "f(V_(i-1))", "f(V_i)", "V_(i+1)", "εa (%)"],
    loc ='center'
)

# Letra E
# Vantagem: a secante não precisa da derivada exata, só aproxima com dois pontos.
# Desvantagem: converge um pouco mais devagar que Newton-Raphson, 
# e diferente da falsa posição, não garante que a raiz continue cercada pelo intervalo (pode divergir)

plt.tight_layout()
plt.show()
