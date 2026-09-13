import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# o intervalo é um vetor de de a até b: [a, b]
# precisão absoluta ε(epsilon)
def bissection(f, a, b, epsilon):
    k = 1
    historico = [] # lista vazia
    x_anterior = None 

    # da pra botar limite de iteração com AND k < limite
    while (b - a) > epsilon: 
        # descobre o x
        x = (a + b) / 2 

        # Calcula erro aproximado (exceto na 1a iteração)
        if x_anterior is None:
            erro_a = None
        else:
            erro_a = abs(x - x_anterior) / abs(x) * 100
            erro_a = round(erro_a, 6) # faz o round aqui pra n usar round com None, q ai da erro

        #guarda os dados da iteração
        historico.append([k, round(a, 4), round(b, 4), round(x, 4), round(f(x), 6), erro_a])

        # avalia f(x) e multiplica por f(a)
        if (f(a)*f(x)) > 0:
            # se f(a).f(x) > 0, então a raiz ta no intervalo [x, b] e não [a, b]
            a = x # ent etualiza o a
        else:
            # se f(a).f(x) < 0, então a raiz ta no intervalo [a, x] e não [a, b]
            b = x # ent atualiza o b

        x_anterior = x # guarda pra calcular o erro relativo percentual
        k = k + 1 #conta quantas iterações o código teve

    # print(k)
    # funciona pq python n tem escopo de bloco, ou seja, while, if, else e afins n criam um escopo novo
    return x, historico

# main
# função que define a velocidade de queda de um paraquedista
# queremos saber 1o, qual o valor de c, que se a gente botar na fórmula vai retornar 10
# v(t) = (gm/c) * (1-e^-(c*t/m)) onde:
    # m = 68,1 kg       (massa do paraquedista)
    # g = 9,81 m/s²     (aceleração da gravidade)
    # após t = 10s      (tempo em queda)
    # v = 40 m/s        (velocidade da queda)
    # c = ?             (coeficiente de resistência do ar) 

# pra usar nossos métodos de aproximação, precisamos transformar isso em um problema de achar a raiz
# ou seja qual valor de de c pra f(c) = v(10) - 40 retornar 0

def f(c):
    m = 68.1
    g = 9.81
    t = 10

    # np.exp já calcula euler ^ alguma coisa. np.exp(alguma coisa)
    # isso aqui é uma inversão matemática da fórmula, a gente quer 0(f(c)) = esse return
    # basicamente botar no papel e trocar os valores
    return (g * m / c) * (1 - np.exp(-c*t/m)) - 40

# região da raiz é +- no c = 14.8
# plotei o grafo, e anotei. 
# Movi o código de plotagem pro final pra ficar + organizado

a = f(10)
b = f(20)
if a * b < 0:
    print(a, b)
    print('sinais opostos, há uma raiz no intervalo')

# Resposta da letra C
# a mudança de sinal é importante pro método da bissecção pois pelo teorema do valor intermediário
# se f é contínua nesse intervalo [a, b] e a e b possuem sinais opostos, então certamente a função corta o eixo x.

resultado, tabela_dados = bissection(f, 10, 20, 0.001)

xs = np.linspace(5, 25, 200)
ys = f(xs)

# plt.subplots(2,1...) cria uma figura com 2 linhas e 1 coluna de subgráficos
# figsize define o tamanho que a figura vai utilizar em polegadas por alguma razão
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(xs, ys, label="f(c) = gm/c * 1-e^(-ct/m) - 40")
ax1.spines["left"].set_position(("data", 15))
ax1.spines["bottom"].set_position("zero")
ax1.spines["right"].set_visible(False)
ax1.spines["top"].set_visible(False)
ax1.legend()

# i = número da iteração
# cl (c lower) = extremo inferior da iteração
# cu (c upper) = extremo superior da iteração
# cr (c root) = ponto médio candidato à ser raiz
# f(cr) = valor da função naquele croot. Retorna o quão próx de 0.
ax2.axis('off')
tabela = ax2.table(
    cellText=tabela_dados,
    colLabels=["i", "cl", "cu", "cr", "f(cr)", "εa (%)"],
    loc='center'
)

# Questão e
# c ~ 14,80 kg/s é o coeficiente de resistência do ar que faz o modelo bater com a velocidade medida (40 m/s aos 10s). 
# Fisicamente, quanto maior o c, mais resistência o ar oferece, e menor é a velocidade do paraquedista
# é esse efeito que o paraquedas maximiza

plt.tight_layout() # ajusta o espaço do gráfico e da tabela automaticamente
plt.show()