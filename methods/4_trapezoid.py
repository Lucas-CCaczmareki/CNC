## implementação da regra dos trapézios, com um único segmento e com múltiplos segmentos (fig 21.9 do chapra)

# o objetivo é calcular a integral de f(x) entre a e b, ou seja, a área embaixo da curva. Quando n dá (ou n compensa) pra achar a primitiva,
# a gente aproxima essa área por figuras que sabemos calcular. Aqui a figura é o trapézio: troca a curva por uma reta entre dois pontos da função
# a área do trapézio é base * média das alturas, e a base aqui é o espaçamento h:
# (em pt claro: calcula a área da curva com um monte de trapézio)

# (a) um único segmento: liga (a, f(a)) com (b, f(b)) com uma reta só, e h = b - a
# I ≈ h * (f0 + f1) / 2

# (b) múltiplos segmentos (regra composta): divide [a, b] em n segmentos iguais de largura h = (b - a) / n e aplica o trapézio em cada um
# os pontos são x_i = a + i h, com i = 0, ..., n, e f_i = f(x_i). Somando as áreas de todos os trapézios:
# I ≈ h/2 * [f0 + 2 (f1 + f2 + ... + f_{n-1}) + fn]
# os pontos de dentro aparecem 2x pq cada um é fim de um trapézio e começo do outro. Só o 1o e o último aparecem 1x (é de onde vem o 2 * f_i no DOFOR)
# quanto mais segmentos, mais a reta de cada trapézio cola na curva e menor o erro. Dobrar o n divide o erro por ~4 (o erro é proporcional a h²)
# a regra dá o valor EXATO se f for uma reta (grau 1), pq aí o trapézio é a própria área. Pra curvas sempre tem erro

# h: largura de cada segmento (espaçamento entre os pontos, que precisa ser igual pra todos)
# f0, f1: valores da função nas duas pontas do segmento (só na trap)
# n: número de segmentos (n + 1 pontos)
# f: vetor com os n + 1 valores da função nos pontos igualmente espaçados, f[i] = f(x_i)
# retorna o valor aproximado da integral

# ps: o pseudocódigo do livro usa f_0 ... f_n, que já é 0-based, então aqui o vetor f usa o mesmo índice e o DOFOR i = 1, n - 1 vira range(1, n)
# ps2: a função recebe os VALORES da função (f[i]), não a função em si. Dá pra usar com uma tabela de dados medidos, sem precisar da expressão de f(x)
# ps3: o DOFOR do livro é inclusivo (vai até n - 1), o range do python n é (vai até n - 2), por isso o range(1, n) pega i = 1 ... n - 1

import numpy as np
import sympy as sp

# (a) um único segmento
def trap(h, f0, f1):
    return h * (f0 + f1) / 2

# (b) múltiplos segmentos
def trapm(h, n, f):
    # copia pra n estragar o vetor original
    f = np.array(f, dtype=float)

    # n segmentos precisam de n + 1 pontos
    if len(f) != n + 1:
        print("o vetor f precisa ter n + 1 valores (n segmentos = n + 1 pontos)")
        return np.nan

    # as pontas entram 1x e os pontos de dentro 2x
    soma = f[0]
    for i in range(1, n):
        soma = soma + 2 * f[i]
    soma = soma + f[n]

    return h * soma / 2

# main
