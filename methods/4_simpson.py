## implementação das regras de 1/3 e 3/8 de Simpson, simples e múltiplas (fig 21.13 do chapra)

# mesmo objetivo da regra dos trapézios (4_trapezoid.py): calcular a integral de f(x) entre a e b (a área embaixo da curva)
# a diferença é que em vez de trocar a curva por retas, aqui troca por polinômios, o que cola muito melhor na curva
# o erro cai bem mais rápido com o n do que no trapézio

## (em pt claro): ao invés de calcular a área dividindo a curva em trapézios
## calcula a área usando várias parábolas (aquelas meia luas)

# (a) 1/3 de Simpson: passa uma parábola (polinômio de grau 2) por 3 pontos (2 segmentos), f0, f1, f2
# I ≈ 2h * (f0 + 4 f1 + f2) / 6   (= h/3 * (f0 + 4 f1 + f2), de onde vem o nome)

# (b) 3/8 de Simpson: passa uma cúbica (polinômio de grau 3) por 4 pontos (3 segmentos), f0, f1, f2, f3
# I ≈ 3h * (f0 + 3 (f1 + f2) + f3) / 8

# (c) 1/3 múltiplo: aplica a 1/3 de Simpson em cada PAR de segmentos e soma. n precisa ser PAR (cada parábola gasta 2 segmentos)
# I ≈ h/3 * [f0 + 4 (f1 + f3 + ... + f_{n-1}) + 2 (f2 + f4 + ... + f_{n-2}) + fn]
# os pontos de índice ímpar são o meio de cada parábola (peso 4), os de índice par de dentro são fim de uma e começo da outra (peso 2)
# dá pra ver o 4 e o 2 alternando no DOFOR i = 1, n - 2, 2 (cada volta pega f_i com 4 e f_{i+1} com 2)

# (d) múltiplo geral: funciona pra n par OU ímpar. Se n é par, só aplica a (c). Se n é ímpar sobra 1 segmento que a 1/3 n cobre,
# então usa a 3/8 nos 3 últimos segmentos (4 últimos pontos) e a (c) nos n - 3 primeiros (que sobram em número par)
# n = 1 é um caso à parte: só tem 1 segmento, nem a 1/3 nem a 3/8 cabem, então cai no trapézio

# h: largura de cada segmento (igual pra todos)
# f0, f1, f2, f3: valores da função nos pontos (só nas simp13 e simp38)
# n: número de segmentos (n + 1 pontos)
# f: vetor com os n + 1 valores da função nos pontos igualmente espaçados, f[i] = f(x_i)
# a, b: limites da integral (só na simpint, que calcula o h sozinha)
# retornam o valor aproximado da integral

# as regras de Simpson dão o valor EXATO pra polinômios até grau 3 (mesmo a 1/3, que usa só uma parábola, pq o erro dela depende da 4a derivada, e a de um cúbico é 0)

# ps: o pseudocódigo do livro usa f_0 ... f_n, que já é 0-based. O DOFOR i = 1, n - 2, 2 (inclusivo) vira range(1, n - 1, 2)
# ps2: a Simp13m do livro faz sum = f(0), mas é o f_0 do vetor (o valor, não a função)
# ps3: o pseudocódigo da simpint n inicializa o sum no ELSE, aqui já começa em 0 pq o python reclama de variável não definida
# ps4: o pseudocódigo da simpint pede a trap (1 segmento), então ela tá copiada aqui (mesma do 4_trapezoid.py)
# ps5: o m da simpint é o número de segmentos que sobram pra 1/3 múltipla, e ela usa os m + 1 primeiros valores do vetor f

import numpy as np
import sympy as sp

# a trapézio de 1 segmento, só pro caso n = 1 da simpint
def trap(h, f0, f1):
    return h * (f0 + f1) / 2

# (a) 1/3 de Simpson, 1 aplicação (2 segmentos)
def simp13(h, f0, f1, f2):
    return 2 * h * (f0 + 4 * f1 + f2) / 6

# (b) 3/8 de Simpson, 1 aplicação (3 segmentos)
def simp38(h, f0, f1, f2, f3):
    return 3 * h * (f0 + 3 * (f1 + f2) + f3) / 8

# (c) 1/3 de Simpson múltipla (n par)
def simp13m(h, n, f):
    # copia pra n estragar o vetor original
    f = np.array(f, dtype=float)

    # cada parábola gasta 2 segmentos, então n ímpar n fecha. A simpint já cuida disso, aqui é só uma proteção
    if n % 2 != 0:
        print("a 1/3 de Simpson múltipla precisa de n par (use a simpint pra n ímpar)")
        return np.nan

    if len(f) < n + 1:
        print("o vetor f precisa ter pelo menos n + 1 valores")
        return np.nan

    # o i anda de 2 em 2 pelos pontos de índice ímpar (meio de cada parábola), f_{i+1} é o ponto de índice par logo depois
    soma = f[0]
    for i in range(1, n - 1, 2):
        soma = soma + 4 * f[i] + 2 * f[i + 1]

    # o último par é diferente: o f_{n-1} é meio (peso 4) mas o f_n é a ponta final (peso 1, e não 2)
    soma = soma + 4 * f[n - 1] + f[n]

    return h * soma / 3

# (d) Simpson múltipla pra qualquer n (par ou ímpar)
def simpint(a, b, n, f):
    f = np.array(f, dtype=float)

    if len(f) != n + 1:
        print("o vetor f precisa ter n + 1 valores (n segmentos = n + 1 pontos)")
        return np.nan

    h = (b - a) / n
    soma = 0.0

    if n == 1:
        # só 1 segmento: n cabe nenhuma das duas, usa o trapézio
        soma = trap(h, f[n - 1], f[n])
    else:
        m = n

        # n ímpar: guarda os 3 últimos segmentos (4 últimos pontos) pra 3/8, e o resto (m = n - 3, que é par) vai pra 1/3 múltipla
        if n % 2 != 0 and n > 1:
            soma = soma + simp38(h, f[n - 3], f[n - 2], f[n - 1], f[n])
            m = n - 3

        # m > 1 pq com n = 3, m = 0 e a 3/8 já cobriu tudo. Aqui m é par (n par ou n - 3 de um n ímpar)
        if m > 1:
            soma = soma + simp13m(h, m, f)

    return soma

# main