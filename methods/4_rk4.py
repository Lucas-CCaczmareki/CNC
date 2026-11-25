## implementação do método de Runge-Kutta de ordem 4 (RK4) pra resolver PVI

# o objetivo é resolver um problema de valor inicial (PVI): y' = f(x, y), y(a) = y0, com x em [a, b]
# quase nunca existe solução analítica, então a gente calcula y só em pontos igualmente espaçados: x_i = a + i*h
# o euler usa só a inclinação do começo do passo (k1). O RK4 mede a inclinação em 4 pontos do passo e tira uma média ponderada:
# k1 = f(x, y)                       inclinação no começo
# k2 = f(x + h/2, y + h/2 * k1)      inclinação no meio, chegando lá com o k1
# k3 = f(x + h/2, y + h/2 * k2)      inclinação no meio de novo, agora corrigida com o k2
# k4 = f(x + h, y + h * k3)          inclinação no fim
# y(i+1) = y(i) + h/6 * (k1 + 2*k2 + 2*k3 + k4)    (pesos 1, 2, 2, 1. É a regra de Simpson aplicada na inclinação)

# erro local O(h^5) e erro global O(h^4): dividir o h por 2 faz o erro cair ~16x
# custo: 4 avaliações de f por passo, e n precisa derivar a f

# f: função f(x, y) do PVI
# a, b: limite inferior e limite superior do intervalo
# y0: valor inicial, y(a) = y0
# m: número de subintervalos (m + 1 pontos)
# h: tamanho do passo, h = (b - a) / m
# vet_x, vet_y: abscissas e solução do PVI (m + 1 posições cada)
# info: 0 = ok, 1 = m inválido (precisa ser >= 1) ou b <= a

# ps: o pseudocódigo é 1-based (VetX(1) = a), python é 0-based. Então vet_x[0] = a e vet_x[m] = b
# ps2: o xt é recalculado como a + i*h em vez de xt + h, pra n acumular erro de arredondamento nas abscissas

import numpy as np

def f(x, y):
    return -x*y

def rk4(f, a, b, y0, m):
    if m < 1 or b <= a:
        print("m precisa ser >= 1 e b > a")
        return np.array([]), np.array([]), 1

    h = (b - a) / m
    xt = a
    yt = y0

    vet_x = np.zeros(m + 1)
    vet_y = np.zeros(m + 1)
    vet_x[0] = xt
    vet_y[0] = yt

    for i in range(1, m + 1):
        # k1: inclinação no começo do passo
        x = xt
        y = yt
        k1 = f(x, y)

        # k2: inclinação no meio, usando o k1 pra andar meio passo
        x = xt + h / 2
        y = yt + h / 2 * k1
        k2 = f(x, y)

        # k3: inclinação no meio de novo, mas andando meio passo com o k2 (o x continua xt + h/2)
        y = yt + h / 2 * k2
        k3 = f(x, y)

        # k4: inclinação no fim do passo, andando o passo inteiro com o k3
        x = xt + h
        y = yt + h * k3
        k4 = f(x, y)

        # média ponderada das 4 inclinações (o meio pesa o dobro)
        xt = a + i * h
        yt = yt + h / 6 * (k1 + 2 * (k2 + k3) + k4)

        vet_x[i] = xt
        vet_y[i] = yt

    return vet_x, vet_y, 0

# main
vet_x, vet_y, info = rk4(f, 0, 1, 1, 10)
print(vet_x)
print()
print(vet_y)
print()
print((-vet_x)*vet_y)