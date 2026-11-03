## implementação do polinômio interpolador de Newton em diferenças divididas (fig 18.7 do chapra)

# dado um conjunto de n+1 pontos (x_i, y_i), o objetivo é estimar o valor da função num x qualquer (xi) que n tá na tabela
# o polinômio é construído termo a termo, cada termo novo soma uma correção em cima do anterior:
# p_n(x) = f[x0] + f[x0,x1](x - x0) + f[x0,x1,x2](x - x0)(x - x1) + ... + f[x0,...,xn](x - x0)...(x - x_{n-1})
# onde f[...] são as diferenças divididas, calculadas recursivamente:
# f[x_i,...,x_{i+j}] = (f[x_{i+1},...,x_{i+j}] - f[x_i,...,x_{i+j-1}]) / (x_{i+j} - x_i)

# vantagem sobre o lagrange: se chegar um ponto novo, n precisa refazer tudo, só calcula mais uma linha da tabela e soma mais um termo
# e como cada termo é a correção do anterior, dá de graça uma estimativa do erro de cada ordem (ea)

# x, y: vetores com os pontos conhecidos (x não precisa estar ordenado nem igualmente espaçado, mas n pode ter x repetido)
# n: grau do polinômio (n + 1 pontos)
# xi: ponto onde a gente quer interpolar
# yint: vetor com a estimativa de cada ordem. yint[k] = valor do polinômio de grau k em xi
# ea: vetor com a estimativa do erro. ea[k-1] = yint[k] - yint[k-1] (o termo que o grau k acrescentou)
# fdd: tabela das diferenças divididas. fdd[i, j] = diferença dividida de ordem j começando no ponto i
# a 1a linha (fdd[0, :]) são os coeficientes do polinômio

# ps: o pseudocódigo é 1-based em alguns lugares, python é 0-based. Aqui já tá tudo 0-based

import numpy as np

def newton_interpolacao(x, y, n, xi):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # tabela das diferenças divididas. Só o triângulo superior esquerdo é usado (fdd[i, j] existe pra i + j <= n)
    fdd = np.zeros((n + 1, n + 1))

    # coluna 0 = os próprios valores de y (diferença dividida de ordem 0)
    for i in range(n + 1):
        fdd[i, 0] = y[i]

    # as outras colunas, j = ordem. Cada coluna usa a anterior, por isso o j é o laço de fora
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            fdd[i, j] = (fdd[i + 1, j - 1] - fdd[i, j - 1]) / (x[i + j] - x[i])

    # monta o polinômio avaliando em xi, 1 termo por vez
    xterm = 1.0 # guarda o produto (xi - x0)(xi - x1)...(xi - x_{order-1}), reaproveita o do termo anterior
    yint = np.zeros(n + 1)
    ea = np.zeros(n)
    yint[0] = fdd[0, 0]

    for order in range(1, n + 1):
        xterm = xterm * (xi - x[order - 1])
        yint2 = yint[order - 1] + fdd[0, order] * xterm
        ea[order - 1] = yint2 - yint[order - 1] # erro estimado = o que esse termo acrescentou
        yint[order] = yint2

    return yint, ea, fdd

# main
# exemplo do chapra (ex 18.4): estimar ln(2) usando os pontos de ln(x) em x = 1, 4, 6, 5 (essa ordem mesmo, n precisa ordenar)
# valor verdadeiro: ln(2) = 0,6931472
# resposta esperada (livro): grau 1 = 0,4620981 | grau 2 = 0,5658444 | grau 3 = 0,6287686
# aqui dá 0,462098 | 0,5658442 | 0,6287674, a diferença na 6a casa é pq o y tá arredondado em 6 dígitos (o livro usa o ln exato)
x = [1, 4, 6, 5]
y = [0, 1.386294, 1.791759, 1.609438]
n = 3
xi = 2

yint, ea, fdd = newton_interpolacao(x, y, n, xi)

print("tabela de diferenças divididas (cada coluna é uma ordem) =\n", fdd)
print("\ngrau  yint  ea  erro verdadeiro")
verdadeiro = np.log(xi)
for k in range(n + 1):
    ea_k = ea[k - 1] if k > 0 else None
    print(k, round(float(yint[k]), 7), None if ea_k is None else round(float(ea_k), 7), round(float(verdadeiro - yint[k]), 7))

# confere o polinômio completo com a lib (np.polyfit com grau n passando por todos os pontos dá o mesmo polinômio, pq ele é único)
coef = np.polyfit(x, y, n)
print("\nyint (grau n) =", yint[n])
print("yint (np.polyfit) =", np.polyval(coef, xi))