# Esse arquivo contém algumas funções básicas da biblioteca numpy
# E alguns métodos básicos pra plotar gráficos com a matplotlib

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = 3

# Funções matemáticas padrão
np.cos(x)
np.exp(x)
np.sqrt(x)

# definir a função f(x) pra funcionar com número único ou array
def f(x):
    return x - 2 * np.cos(x)

def f2(x):
    return x - 2 * sp.cos(x) # precisa usar as funções matemáticas da sympy pra ela funcionar

f(1.5) #avaliaçao da f(x) num ponto específico

# a lib sympy utiliza 2 conceitos centrais, os symbols e as expressões
# ela trata eles como se fossem símbolos no papel.

x = sp.symbols('x')                 # guarda dentro de X o symbol X da lib
f_linha_expr = sp.diff(f2(x), x)    # guarda dentro de f_linha a expressão textual da derivada de f()
print(f_linha_expr)                 # isso teoricamente printa a expressão escrita.

f_linha = sp.lambdify(x, f_linha_expr, "numpy")     # lambdify converte a expressão da lib pra uma função python "normal"
print(f_linha(1.5))                                 # avalia a derivada de f num ponto específico

# xs e ys é o "plural" das letras. Padrão utilizado quando nos referimos à arrays de valores pra x e y
# np.linspace chama a biblioteca numpy e gera 200 pontos igualmente espaçados entre -4 e 4.
# xs é o array desses pontos, fica +- [-4, -3.96, ..., 3.96, 4]
xs = np.linspace(-4, 4, 200)

# isso faz o ys virar um array, onde pra cada x, ele aplica a função e guarda o resultado.
# Isso funciona por que as operações do numpy podem ser vetorizadas (funciona recebendo tanto número único quanto vetor de elementos)
ys = f(xs) 

# versão + simples pra testes
# isso faz o valor de y espelhar o de x, ai o gráfico fica uma reta. 
# ys = xs 

# Cria os dois objetos principais da matplotlib, uma figure e um axel (aŕea de desenho)
fig, ax = plt.subplots()

# isso aqui recebe 2 arrays e liga eles ponto a ponto.
ax.plot(xs, ys, label="f(x) = x - 2cos(x)")
ax.plot(xs, xs, label = "fx(x) = x")

# move os eixos x e y pro centro (posição 0)
# por padrão da biblioteca o eixo y é a "borda" esquerda e o x a "borda" inferior
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")

# remove as bordas de cima e da direita
# já que por padrão a biblioteca desenha uma caixa ao entorno do gráfico
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# coloca as setas nas pontas dos eixos (estético). 
# Não entendi esse transform btw. Só entendi que ele faz as referências do x, y nos 1os parametros
# apontarem pro extremo do gráfico sempre
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

ax.legend() # habilita a legenda atribuída ao axel
plt.show() # mostra a figure com o axel que a gente criou

# pra entender como utilizar tabelas simples com o matplot, consultar relatorio_1 -> ex1