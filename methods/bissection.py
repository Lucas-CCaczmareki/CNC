# implementação do algoritmo do método da bissecção
# esse método faz parte dos métodos que quebra.
# queremos achar um x que zere a f(x), essa é uma "raiz"
# recebemos um intervalor [a, b] que contém exatamente uma raiz

# o método da bissecção divide esse intervalo na metade e vai fechando iterativamente o intervalo
# até alcançar a tolerância de erro desejada.

import numpy as np
import matplotlib.pyplot as plt

# o intervalo é um vetor de de a até b: [a, b]
# precisão absoluta ε(epsilon)

def f(x):
    # função do slide, nosso objetivo é achar o x que faz o return ser = 0
    return x - 2*np.cos(x) 

def bissection(a, b, epsilon):
    k = 1

    # da pra botar limite de iteração com AND k < limite
    while (b - a) > epsilon: 
        # descobre o x
        x = (a + b) / 2 

        # avalia f(x) e multiplica por f(a)
        if (f(a)*f(x)) > 0:
            # se f(a).f(x) > 0, então a raiz ta no intervalo [x, b] e não [a, b]
            a = x # ent etualiza o a
        else:
            # se f(a).f(x) < 0, então a raiz ta no intervalo [a, x] e não [a, b]
            b = x # ent atualiza o b

        #conta quantas iterações o código teve
        k = k + 1 

    print(k)
    # funciona pq python n tem escopo de bloco, ou seja, while, if, else e afins n criam um escopo novo
    return x 

# main
result = bissection(0, 2, 0.001)
print(result)

xs = np.linspace(-5, 5, 200)
ys = f(xs) # a minha raiz ta no intervalo arbitrário [0, 2]

fig, ax = plt.subplots()
ax.plot(xs, ys, label = "f(x) = x - 2*np.cos(x)")

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# f"..." são as fstrings em python, pra colocar uma variável dentro de um bloco de texto automaticamente
# marca e escreve um texto perto do ponto x que zera a f(x)
ax.scatter(result, 0, color="red", zorder=5, label=f"raiz = {result:.4f}") #zorder define o quão pra cima nas camadas de imagem o ponto vai ficar
ax.annotate(f"x = {result:.4f}",
            xy=(result, 0),         # ponto que a seta da caixa de texto aponta
            xytext=(2, 1.5),   # onde o texto vai ficar
            arrowprops=dict(arrowstyle="->"))

ax.legend()
plt.show()


