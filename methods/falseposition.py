# implementação do algoritmo do método da falsa posição
# esse método faz parte dos métodos que quebra.
# queremos achar um x que zere a f(x), essa é uma "raiz"
# recebemos um intervalor [a, b] que contém exatamente uma raiz

# calcula uma reta secante que corta os pontos (a, f(a)) e (b, f(b))
# o lugar onde essa reta corta o x é o candidato pra reduzir o intervalo

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    # função do slide, nosso objetivo é achar o x que faz o return ser = 0
    return x - 2*np.cos(x) 

# recebe um intervalo [a, b]
# epsilon1: mede o tamanho do intervalo (b - a)
# epsilon2: mede |f(x)|, que é o quão perto de zero a f(x) já está

def falsepos(a, b, epsilon1, epsilon2):
    k = 1

    # se o intervalo já é pequeno o suficiente, retorna o x
    if (b - a) < epsilon1: return (a*f(b) - b*f(a)) / (f(b) - f(a))

    # se algum dos extremos já está suficientemente perto de zero, retorna eles como resultado
    if abs(f(a)) < epsilon2:
        return a
    else: 
        if abs(f(b)) < epsilon2: return b

    while (b - a) > epsilon1: 
        # calcula o ponto em x que a reta secante corta
        x = (a*f(b) - b*f(a)) / (f(b) - f(a)) # não entendo entendo a fórmula mas em teoria ela é um padrão
        if abs(f(x)) < epsilon2: return x

        if (f(a)*f(x)) > 0:
            a = x # ent etualiza o a
        else:
            b = x # ent atualiza o b
        
        #conta quantas iterações o código teve
        k = k + 1 

    print(k)
    return x

result = falsepos(0, 2, 0.001, 0.001)
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