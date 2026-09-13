import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# recebe um intervalo [a, b]
# epsilon1: mede o tamanho do intervalo (b - a)
# epsilon2: mede |f(x)|, que é o quão perto de zero a f(x) já está
def falsepos(f, a, b, epsilon1, epsilon2):
    k = 1
    historico = []
    x_anterior = None

    # se o intervalo já é pequeno o suficiente, retorna o x
    if (b - a) < epsilon1: return (a*f(b) - b*f(a)) / (f(b) - f(a)), historico

    # se algum dos extremos já está suficientemente perto de zero, retorna eles como resultado
    if abs(f(a)) < epsilon2:
        return a, historico
    else: 
        if abs(f(b)) < epsilon2: return b, historico

    while (b - a) > epsilon1: 
        # calcula o ponto em x que a reta secante corta
        x = (a*f(b) - b*f(a)) / (f(b) - f(a)) 

        if x_anterior is None:
            erro_a = None
        else:
            erro_a = abs(x - x_anterior) / abs(x) * 100
            erro_a = round(erro_a, 6)

        # guarda dados pra resposta da letra c
        historico.append([k, round(a, 4), round(b, 4), round(x, 4), round(f(x), 6), erro_a])

        if abs(f(x)) < epsilon2: return x, historico

        if (f(a)*f(x)) > 0:
            a = x # ent etualiza o a
        else:
            b = x # ent atualiza o b
        
        #conta quantas iterações o código teve
        x_anterior = x
        k = k + 1 

    # print(k)
    return x, historico

# main
# vazão via equação de manning
# Q = 1/n * A * Rh^(2/3) * S^(1/2)
    # Q = vazão
    # A = área molhada = b * y
        # b = largura
        # y = profundidade da água
    # Rh = raio hidráulico = b*y / (b + 2*y)
    # S = declividade do canal
    # n = coeficiente de rugosidade

# y que zera a função é ~1.566 (conferindo no gráfico)
def f(y):
    n = 0.025
    b = 3
    s = 0.001
    q = 5
    return ((1/n) * (b*y) * (np.power((b*y)/(b+2*y), 2/3) * s**0.5)) - q

resultado, tabela_dados = falsepos(f, 0.2, 3, 0.001, 0.001)

xs = np.linspace(0.2, 3, 200)
ys = f(xs)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(xs, ys, label="f(y) = ((1/n) * (b*y) * (np.power((b*y)/(b+2*y), 2/3) * s**0.5)) - q")
ax1.spines["left"].set_position(("data", 1.75))
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_position("zero")
ax1.spines["top"].set_visible(False)
ax1.legend()

ax2.axis('off')
# útil guardar a tabela numa variável se eu quiser alterar algo dps
# pode acabar dando problema nesse método por q ele pode retornar tabela_dados vazia. Mas pra esse caso específico ta de boa
tabela = ax2.table(
    cellText = tabela_dados,
    colLabels = ["i", "yl", "yu", "cr", "f(cr)", "εa (%)"],
    loc = 'center'
)

# Questão D
# por que o método da falsa posição estima a posição do próximo candidato de maneira diferente
# do método da bissecção. Enquanto a bissecção divide o intervalo sempre no meio, a falsa posição
# usa uma reta secante e pega onde o y=0, então se f(a) tá mais perto 0 q f(b) por exemplo, o
# ponto calculado fica mais próximo de a, e vice-versa. 
# Dai surge a diferença nas aproximações. Um pega sempre pela metade, o outro pondera quem ta mais perto.

# Questão E
y_metros = resultado
y_cm = y_metros * 100
print(y_metros, y_cm)

# y ~ 1,57 m (157,18 cm) é a profundidade mínima de água necessária pro canal escoar 5,0 m³/s. 
# Como é mais da metade da largura do canal (3,0 m), indica um canal relativamente profundo
# na prática, ainda se somaria uma margem de segurança acima desse valor.

plt.tight_layout()
plt.show()