## implementação do algoritmo de spline cúbica natural (chapra, cap 18)

# em vez de ajustar 1 polinômio de grau alto em todos os pontos (que oscila), ajusta 1 cúbica por intervalo
# e emenda elas de um jeito que a curva, a 1a derivada e a 2a derivada sejam contínuas nos nós
# S_i(x) = d2x_(i-1)/(6h) (x_i - x)³ + d2x_i/(6h) (x - x_(i-1))³ + c3 (x_i - x) + c4 (x - x_(i-1))   com h = x_i - x_(i-1)
# as únicas incógnitas são as segundas derivadas nos nós (d2x). Impondo a continuidade da 1a derivada
# em cada nó interno sai um sistema TRIDIAGONAL pras d2x, e a condição "natural" fecha o sistema: d2x_0 = d2x_n = 0

# x, y: pontos (x0, y0) ... (xn, yn). Os x tem que estar em ordem crescente
# xu: ponto onde a gente quer interpolar
# n: número de intervalos (n+1 pontos)
# e, f, g, r: diagonal inferior, principal, superior e termos independentes do sistema tridiagonal
# d2x: segundas derivadas da spline em cada nó (as incógnitas)
# yu, dy, d2y: valor da spline, da 1a derivada e da 2a derivada em xu
# info: 0 = ok, 1 = xu fora do intervalo [x0, xn] (yu, dy e d2y voltam nan)

# ps: o pseudocódigo do chapra já é 0-based pro x e y (x0 ... xn), então eles ficam iguais ao livro
# ps2: o sistema tridiagonal só tem as linhas 1 até n-1 (os nós internos), por isso e, f, g, r e d2x são
# alocados com tamanho n+1 e usam o MESMO índice do livro (a posição 0 de e, f, g, r fica sem uso)
# ps3: o decomp e o subst são a eliminação de gauss pra matriz tridiagonal (algoritmo de thomas), custa O(n)
# ps4: precisa de pelo menos 3 pontos (n >= 2)

import numpy as np

# monta o sistema tridiagonal. A primeira e a última linha são diferentes pq d2x_0 = d2x_n = 0 (natural)
def tridiag(x, y, n):
    e = np.zeros(n + 1)
    f = np.zeros(n + 1)
    g = np.zeros(n + 1)
    r = np.zeros(n + 1)

    # primeira linha (nó 1): n tem e, pq ele multiplicaria o d2x_0 que é 0
    f[1] = 2 * (x[2] - x[0])
    g[1] = x[2] - x[1]
    r[1] = 6 / (x[2] - x[1]) * (y[2] - y[1])
    r[1] = r[1] + 6 / (x[1] - x[0]) * (y[0] - y[1])

    # linhas do meio (nós 2 até n-2)
    for i in range(2, n - 1):
        e[i] = x[i] - x[i - 1]
        f[i] = 2 * (x[i + 1] - x[i - 1])
        g[i] = x[i + 1] - x[i]
        r[i] = 6 / (x[i + 1] - x[i]) * (y[i + 1] - y[i])
        r[i] = r[i] + 6 / (x[i] - x[i - 1]) * (y[i - 1] - y[i])

    # última linha (nó n-1): n tem g, pq ele multiplicaria o d2x_n que é 0
    # com n = 2 só existe 1 incógnita, a primeira linha já é o sistema inteiro (se rodasse isso o r[1] seria somado 2x)
    if n > 2:
        e[n - 1] = x[n - 1] - x[n - 2]
        f[n - 1] = 2 * (x[n] - x[n - 2])
        r[n - 1] = 6 / (x[n] - x[n - 1]) * (y[n] - y[n - 1])
        r[n - 1] = r[n - 1] + 6 / (x[n - 1] - x[n - 2]) * (y[n - 2] - y[n - 1])

    return e, f, g, r

# decomposição LU do sistema tridiagonal (sem pivotação). Escreve em cima de e e f
# o e vira o multiplicador (L) e o f vira a diagonal da U. O g n muda
def decomp(e, f, g, n):
    for k in range(2, n + 1):
        e[k] = e[k] / f[k - 1]
        f[k] = f[k] - e[k] * g[k - 1]

# resolve o sistema já decomposto. Escreve a solução em x e estraga o r
def subst(e, f, g, r, n, x):
    # substituição progressiva (aplica o L no r)
    for k in range(2, n + 1):
        r[k] = r[k] - e[k] * r[k - 1]

    # substituição retroativa (resolve a U)
    x[n] = r[n] / f[n]
    for k in range(n - 1, 0, -1):
        x[k] = (r[k] - g[k] * x[k + 1]) / f[k]

# acha o intervalo [x_(i-1), x_i] que tem o xu e avalia a cúbica dele (e as derivadas)
def interpol(x, y, n, d2x, xu):
    flag = 0
    i = 1
    yu = dy = d2y = np.nan

    while True:
        if xu >= x[i - 1] and xu <= x[i]:
            # coeficientes da cúbica desse intervalo
            c1 = d2x[i - 1] / 6 / (x[i] - x[i - 1])
            c2 = d2x[i] / 6 / (x[i] - x[i - 1])
            c3 = y[i - 1] / (x[i] - x[i - 1]) - d2x[i - 1] * (x[i] - x[i - 1]) / 6
            c4 = y[i] / (x[i] - x[i - 1]) - d2x[i] * (x[i] - x[i - 1]) / 6

            # valor da spline
            t1 = c1 * (x[i] - xu) ** 3
            t2 = c2 * (xu - x[i - 1]) ** 3
            t3 = c3 * (x[i] - xu)
            t4 = c4 * (xu - x[i - 1])
            yu = t1 + t2 + t3 + t4

            # 1a derivada
            t1 = -3 * c1 * (x[i] - xu) ** 2
            t2 = 3 * c2 * (xu - x[i - 1]) ** 2
            t3 = -c3
            t4 = c4
            dy = t1 + t2 + t3 + t4

            # 2a derivada (é uma reta, vai de d2x_(i-1) até d2x_i)
            t1 = 6 * c1 * (x[i] - xu)
            t2 = 6 * c2 * (xu - x[i - 1])
            d2y = t1 + t2

            flag = 1
        else:
            i = i + 1

        if i == n + 1 or flag == 1:
            break

    if flag == 0:
        print("fora do intervalo")
        return yu, dy, d2y, 1

    return yu, dy, d2y, 0

def spline(x, y, xu):
    # copia pra n estragar os vetores originais
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1 # n intervalos = n+1 pontos

    if n < 2:
        raise ValueError("a spline cúbica precisa de pelo menos 3 pontos")

    d2x = np.zeros(n + 1) # d2x[0] = d2x[n] = 0 (condição natural), os de dentro o subst preenche

    e, f, g, r = tridiag(x, y, n)
    decomp(e, f, g, n - 1)        # o sistema só tem n-1 incógnitas (nós internos)
    subst(e, f, g, r, n - 1, d2x)
    yu, dy, d2y, info = interpol(x, y, n, d2x, xu)

    return yu, dy, d2y, d2x, info

# main
# exemplo do chapra: pontos (3; 2,5), (4,5; 1), (7; 2,5), (9; 0,5), interpolar em xu = 5
# resposta esperada: S(5) ≈ 1,1029
x = [3, 4.5, 7, 9]
y = [2.5, 1, 2.5, 0.5]

yu, dy, d2y, d2x, info = spline(x, y, 5)

print("d2x =", d2x)
print("S(5) =", yu)
print("S'(5) =", dy)
print("S''(5) =", d2y)
print("info =", info)

# confere com a lib (bc_type='natural' é a mesma condição d2x_0 = d2x_n = 0)
try:
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y, bc_type='natural')
    print("S(5) (scipy) =", cs(5))
    print("S'(5) (scipy) =", cs(5, 1))
    print("S''(5) (scipy) =", cs(5, 2))
except ImportError:
    print("scipy n instalado, pulei a conferência")

# xu fora do intervalo [3, 9]. Tem que avisar e voltar info = 1
print("\nteste fora do intervalo:")
_, _, _, _, info2 = spline(x, y, 10)
print("info =", info2)