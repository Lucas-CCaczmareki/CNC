# relatório 3 - exercício 3: perfil topográfico, spline cúbica natural
# usa as funções spline e coeficientes_spline (methods/3_natural_cubic_spline.py) e lagrange (methods/3_lagrange_interpolation.py, só pra comparação)
# copiadas aqui pra esse arquivo rodar sozinho

import numpy as np
import matplotlib.pyplot as plt

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

# converte as d2x (segundas derivadas nos nós) pros coeficientes de cada intervalo, na forma que o enunciado pede:
#   S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)² + d_i (x - x_i)³,   x_i <= x <= x_(i+1)
# derivando a cúbica do chapra em x_i, com h = x_(i+1) - x_i:
#   a_i = y_i | b_i = (y_(i+1) - y_i)/h - h (2 d2x_i + d2x_(i+1))/6 | c_i = d2x_i/2 | d_i = (d2x_(i+1) - d2x_i)/(6h)
# retorna uma matriz n x 4, onde a linha i = [a_i, b_i, c_i, d_i] do intervalo [x_i, x_(i+1)]
def coeficientes_spline(x, y, d2x):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x) - 1

    coef = np.zeros((n, 4))
    for i in range(n):
        h = x[i + 1] - x[i]
        coef[i, 0] = y[i]
        coef[i, 1] = (y[i + 1] - y[i]) / h - h * (2 * d2x[i] + d2x[i + 1]) / 6
        coef[i, 2] = d2x[i] / 2
        coef[i, 3] = (d2x[i + 1] - d2x[i]) / (6 * h)

    return coef

def lagrange(x, y, n, xi):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    soma = 0.0
    for i in range(n + 1):
        # começa com y_i e vai multiplicando pelos fatores de L_i(xi)
        produto = y[i]
        for j in range(n + 1):
            # o fator j = i é pulado (dava 0/0, e a definição do L_i já exclui ele)
            if i != j:
                produto = produto * (xi - x[j]) / (x[i] - x[j])
        soma = soma + produto

    return soma

# imprime uma tabela simples: colunas = lista de nomes, linhas = lista de listas
def imprimir_tabela(colunas, linhas):
    print(" | ".join(f"{c:>12}" for c in colunas))
    print("-" * (15 * len(colunas)))
    for linha in linhas:
        print(" | ".join(f"{'-':>12}" if v is None else f"{v:>12.6g}" for v in linha))

# main
x = [0, 80, 170, 290, 420, 600]            # m (distância)
z = [12.0, 14.8, 13.9, 18.1, 17.4, 20.2]   # m (elevação)
n = len(x) - 1                             # número de intervalos (6 pontos -> 5 intervalos)

# ---------------------------------------------------------------- letra a) coeficientes de cada intervalo
# o spline() devolve as segundas derivadas nos nós (d2x). A partir delas saem os coeficientes de cada intervalo
_, _, _, d2x, _ = spline(x, z, x[0])
coef = coeficientes_spline(x, z, d2x)

print("a) segundas derivadas nos nós (d2x) =", d2x)
print("\nS_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)² + d_i (x - x_i)³ em [x_i, x_(i+1)]")
imprimir_tabela(["i", "x_i", "x_(i+1)", "a_i", "b_i", "c_i", "d_i"], [[i, x[i], x[i + 1], *coef[i]] for i in range(n)])

# conferências
# 1) a forma a + b dx + c dx² + d dx³ tem que dar o mesmo valor que o spline() (forma do chapra) em qualquer ponto
def avaliar(coef, x, xu):
    i = min(np.searchsorted(x, xu, side="right") - 1, len(coef) - 1)
    dx = xu - x[i]
    return coef[i, 0] + coef[i, 1] * dx + coef[i, 2] * dx**2 + coef[i, 3] * dx**3

xg = np.linspace(0, 600, 200)
print("\n|forma a,b,c,d - spline()| máx em 200 pontos =", max(abs(avaliar(coef, x, xu) - spline(x, z, xu)[0]) for xu in xg))

# 2) continuidade da curva, da 1a e da 2a derivada nos nós internos (S_i(x_(i+1)) = S_(i+1)(x_(i+1)) etc)
c0, c1, c2 = 0.0, 0.0, 0.0
for i in range(n - 1):
    h = x[i + 1] - x[i]
    a_i, b_i, c_i, d_i = coef[i]
    c0 = max(c0, abs(a_i + b_i * h + c_i * h**2 + d_i * h**3 - coef[i + 1, 0]))
    c1 = max(c1, abs(b_i + 2 * c_i * h + 3 * d_i * h**2 - coef[i + 1, 1]))
    c2 = max(c2, abs(2 * c_i + 6 * d_i * h - 2 * coef[i + 1, 2]))
print("descontinuidade máx nos nós internos: S =", c0, "| S' =", c1, "| S'' =", c2)

# 3) condição natural: S'' = 0 nas pontas
h_fim = x[n] - x[n - 1]
print("S''(x_0) =", 2 * coef[0, 2], "| S''(x_n) =", 2 * coef[n - 1, 2] + 6 * coef[n - 1, 3] * h_fim)

# 4) passa nos pontos medidos
err_ex3 = max(abs(spline(x, z, xk)[0] - zk) for xk, zk in zip(x, z))
print("passa nos pontos? max|S(x_i) - z_i| =", err_ex3)

# 5) confere com a lib (bc_type='natural' é a mesma condição d2x_0 = d2x_n = 0)
try:
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, z, bc_type='natural')
    print("max|S - scipy| em 200 pontos =", np.max(np.abs(np.array([spline(x, z, xu)[0] for xu in xg]) - cs(xg))))
except ImportError:
    print("scipy n instalado, pulei a conferência")
# Resposta da letra a: os coeficientes de cada intervalo estão na tabela acima, na forma S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)² + d_i (x - x_i)³ em [x_i, x_(i+1)]. O a_i é só a elevação do nó da esquerda, e c_0 = 0 (a condição natural).
#
# As conferências batem: a forma a,b,c,d dá o mesmo valor que o spline() (~1e-14), a curva, a 1ª e a 2ª derivada são contínuas nos nós internos (descontinuidade ~1e-15), S'' = 0 nas duas pontas e o resultado coincide com o CubicSpline do scipy. As segundas derivadas nos nós ficam na ordem de 1e-3 ou menos, pq o perfil é suave.

# ---------------------------------------------------------------- letra b) elevações em x = 125, 230 e 500 m
x_est = [125, 230, 500]
z_est = []
print("b) elevações estimadas")
for xu in x_est:
    yu, dy, d2y, _, info = spline(x, z, xu)
    z_est.append(yu)
    # só pra comparar: interpolação linear entre os 2 vizinhos e o polinômio de grau 5 por todos os pontos (lagrange)
    print(f"  z({xu}) = {yu:.4f} m | S'({xu}) = {dy:.5f} ({100 * dy:.2f} % de inclinação) | S'' = {d2y:.2e} | info = {info}")
    print(f"      linear entre vizinhos = {np.interp(xu, x, z):.4f} | polinômio de grau 5 = {lagrange(x, z, n, xu):.4f}")
# Resposta da letra b: z(125) ≈ 14,335 m, z(230) ≈ 15,759 m e z(500) ≈ 17,853 m.
#
# - A inclinação do terreno (S') em cada ponto: -1,83 % em 125 m (descendo, entre o pico de 14,8 m e o vale de 13,9 m), +4,49 % em 230 m (subida mais forte, entre 170 e 290 m) e +1,64 % em 500 m (subida suave).
# - Pra comparar, a interpolação linear entre os vizinhos dá 14,350 / 16,000 / 18,644 m e o polinômio de grau 5 por todos os pontos dá 13,994 / 15,534 / 11,692 m. Em x = 500 m a spline (17,853) fica abaixo da linear porque herda a tendência de descida dos trechos anteriores (S'(420) ≈ -0,0073, ainda descendo) e só sobe depois, enquanto a linear liga direto 17,4 a 20,2 (o intervalo [420, 600] tem 180 m, o maior). Já o polinômio de grau 5 dá uma elevação absurda em 500 m (11,7 m, bem abaixo dos 17,4 m e 20,2 m dos vizinhos), veja a letra c.

# ---------------------------------------------------------------- letra c) perfil topográfico
xs = np.linspace(0, 600, 300)
zs = [interpol(np.array(x, dtype=float), np.array(z, dtype=float), n, d2x, xu)[0] for xu in xs] # reaproveita as d2x

# só de comparação: o polinômio de grau 5 pelos 6 pontos (lagrange)
zp = [lagrange(x, z, n, xu) for xu in xs]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs, zp, "--", color="gray", label="polinômio de grau 5 (só comparação)")
ax.plot(xs, zs, label="spline cúbica natural")
ax.plot(x, z, "o", color="black", label="dados medidos")
ax.plot(x_est, z_est, "s", color="red", label="estimativas")
# deslocamento de cada rótulo (o do 125 vai pra baixo e pra esquerda, pq pra direita ele bate no ponto medido de 170 m)
desloc = [(-78, -22), (8, -16), (8, -16)]
for xe, ze, d in zip(x_est, z_est, desloc):
    ax.annotate(f"({xe}; {ze:.2f})", (xe, ze), textcoords="offset points", xytext=d)
ax.set_xlabel("distância x (m)")
ax.set_ylabel("elevação z (m)")
ax.set_title("Perfil topográfico da estrada")
ax.legend()
ax.grid(True)
plt.show()

print(f"elevação mínima entre 420 e 600 m: spline = {min(z_ for x_, z_ in zip(xs, zs) if x_ >= 420):.2f} m | polinômio de grau 5 = {min(z_ for x_, z_ in zip(xs, zp) if x_ >= 420):.2f} m")
# Resposta da letra c: o perfil sobe de 12 m pra 20,2 m ao longo dos 600 m, com um pequeno morro em 80 m, um vale em 170 m, uma subida mais forte até 290 m, uma leve descida até 420 m e a subida final até 600 m. A spline passa em todos os pontos medidos, e os 3 pontos estimados (vermelho) ficam sobre a curva, entre os nós.
#
# A curva cinza tracejada (polinômio de grau 5 pelos mesmos 6 pontos) é só pra comparar: ela passa nos pontos, mas oscila entre os nós e afunda pra ~10,7 m entre 420 e 600 m, abaixo de todas as elevações medidas ali (17,4 e 20,2 m), sem respaldo nenhum nos dados. A spline fica entre ~17,3 e 20,2 m nesse trecho, pq cada intervalo é uma cúbica separada: por isso a spline é a escolha melhor pra um perfil com vários pontos.
