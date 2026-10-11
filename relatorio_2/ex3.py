# relatório 2 - exercício 3: duas tubulações em paralelo, F(q1, q2, h) = 0 por Newton
# newton adaptado de methods/newton.py com 3 mudanças que o enunciado pede:
#   1) a jacobiana vem da mão (analítica), e a gente confere ela com a do sympy
#   2) cada iteração resolve J(x_k) Δx = -F(x_k) com a nossa fatoração LU (lu / resolver_lu de methods/lu.py)
#   3) critério de parada: ||Δx||inf / max(||x(k+1)||inf, 1) < 1e-6  OU  ||F(x(k+1))||inf < 1e-8

import numpy as np
import sympy as sp

def lu(A):
    # copia pra n estragar a matriz original (o algoritmo escreve dentro da própria A)
    A = np.array(A, dtype=float)
    m, n = A.shape

    pivot = np.arange(m) # começa [0, 1, 2, ..., m-1] (sem nenhuma troca)
    pdu = 1.0
    info = 0

    for j in range(min(m, n)):
        # escolha do pivô: procura na coluna j (da diagonal pra baixo) o maior elemento em módulo
        p = j
        amax = abs(A[j, j])
        for k in range(j + 1, m):
            if abs(A[k, j]) > amax:
                amax = abs(A[k, j])
                p = k

        # se o maior não tá na diagonal, troca a linha j com a linha p (a linha inteira, incluindo os multiplicadores já guardados)
        if p != j:
            for k in range(n):
                A[j, k], A[p, k] = A[p, k], A[j, k]
            pivot[j], pivot[p] = pivot[p], pivot[j] # registra a troca
            pdu = -pdu # cada troca de linha inverte o sinal do determinante

        pdu = pdu * A[j, j]

        # eliminação de Gauss
        if abs(A[j, j]) != 0:
            r = 1 / A[j, j]
            for i in range(j + 1, m):
                mult = A[i, j] * r # multiplicador m_ij
                A[i, j] = mult     # guarda ele no lugar do zero (essa parte é o L)
                for k in range(j + 1, n):
                    A[i, k] = A[i, k] - mult * A[j, k]
        else:
            # pivô zero = matriz singular. Guarda só o primeiro que aparecer
            if info == 0:
                info = j + 1

    # A agora é U + L - I (U na diagonal e acima, multiplicadores do L abaixo da diagonal)
    return A, pivot, pdu, info

# separa a matriz decomposta em L e U, só pra poder ver/conferir
def extrair_lu(A_lu):
    m, n = A_lu.shape
    k = min(m, n)
    L = np.tril(A_lu[:, :k], -1) + np.eye(m, k) # abaixo da diagonal + 1s na diagonal
    U = np.triu(A_lu[:k, :])                    # diagonal e acima
    return L, U

# resolve Ax = b usando a decomposição (só funciona pra A quadrada e não singular)
def resolver_lu(A_lu, pivot, b):
    n = A_lu.shape[0]
    pb = np.array(b, dtype=float)[pivot] # P b: aplica as mesmas trocas de linha no b

    # L y = P b  (substituição progressiva, de cima pra baixo)
    y = np.zeros(n)
    for i in range(n):
        y[i] = pb[i] - np.dot(A_lu[i, :i], y[:i]) # a diagonal de L é 1, n precisa dividir

    # U x = y  (substituição retroativa, de baixo pra cima)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(A_lu[i, i+1:], x[i+1:])) / A_lu[i, i]

    return x

# imprime uma tabela simples: colunas = lista de nomes, linhas = lista de listas
def imprimir_tabela(colunas, linhas):
    print(" | ".join(f"{c:>12}" for c in colunas))
    print("-" * (15 * len(colunas)))
    for linha in linhas:
        print(" | ".join(f"{'-':>12}" if v is None else f"{v:>12.6g}" for v in linha))

# F e J numéricas. x = [q1, q2, h]
def F(x):
    q1, q2, h = x
    return np.array([q1 + q2 - 0.120,
                     h - 1200 * q1**2,
                     h - 2500 * q2**2])

# letra a) jacobiana analítica: J[i][j] = dFi/dxj, com x = (q1, q2, h)
#   F1 = q1 + q2 - 0,120      -> dF1/dq1 = 1        dF1/dq2 = 1        dF1/dh = 0
#   F2 = h - 1200 q1²         -> dF2/dq1 = -2400 q1 dF2/dq2 = 0        dF2/dh = 1
#   F3 = h - 2500 q2²         -> dF3/dq1 = 0        dF3/dq2 = -5000 q2 dF3/dh = 1
def J(x):
    q1, q2, h = x
    return np.array([[1.0,        1.0,         0.0],
                     [-2400 * q1, 0.0,         1.0],
                     [0.0,        -5000 * q2,  1.0]])

# confere a jacobiana da mão com a do sympy num ponto qualquer (se der ~0 a mão tá certa)
sq1, sq2, sh = sp.symbols('q1 q2 h')
F_sym = sp.Matrix([sq1 + sq2 - sp.Rational(12, 100), sh - 1200*sq1**2, sh - 2500*sq2**2])
J_sym = sp.lambdify([sq1, sq2, sh], F_sym.jacobian([sq1, sq2, sh]), "numpy")
ponto = [0.07, 0.05, 6.0]
print("a) J analítica no ponto", ponto, "=\n", J(ponto))
print("   |J analítica - J sympy| =", np.max(np.abs(J(ponto) - np.array(J_sym(*ponto)))))

def newton(F, J, x0, toler_rel, toler_res, k_max):
    x = np.array(x0, dtype=float)
    k = 0
    info = 0
    historico = [] # [k, x1..xn, Δx1..Δxn, dr, ||F||]
    # k = 0 só mostra o ponto inicial
    historico.append([0, *x, *[None]*len(x), None, np.max(np.abs(F(x)))])

    while True:
        # monta J(x_k) e F(x_k), fatora J e resolve J Δx = -F  (letra c)
        J_lu, pivot, pdu, info_lu = lu(J(x))
        if info_lu != 0:
            print("jacobiana singular, escolha outro x0")
            info = 2
            break
        dx = resolver_lu(J_lu, pivot, -F(x))

        x = x + dx
        k = k + 1
        dr = np.max(np.abs(dx)) / max(np.max(np.abs(x)), 1)
        norma_f = np.max(np.abs(F(x)))
        historico.append([k, *x, *dx, dr, norma_f])

        # os critérios aqui são alternativos (OU)
        if dr < toler_rel or norma_f < toler_res:
            break
        if k >= k_max:
            info = 1
            break

    return x, k, info, historico

# main
x0 = [0.060, 0.060, 5.0]
x, k, info, historico = newton(F, J, x0, 1e-6, 1e-8, 100)

print("\nb) e c) Newton, x0 =", x0)
imprimir_tabela(["k", "q1", "q2", "h", "Δq1", "Δq2", "Δh", "dr", "||F||"], historico)
print("x =", x, "| iterações =", k, "| info =", info)

# verificação: as duas perdas de carga têm que ser iguais a h, e as vazões somam 0,120
print("\nverificação: q1 + q2 =", x[0] + x[1], "| K1 q1² =", 1200 * x[0]**2, "| K2 q2² =", 2500 * x[1]**2, "| h =", x[2])
print("fração da vazão: tubo 1 =", x[0] / 0.120, "| tubo 2 =", x[1] / 0.120)
print("x (np.linalg.solve em cada iteração, só pra conferir) :", end=" ")
xc = np.array(x0, dtype=float)
for _ in range(10):
    xc = xc + np.linalg.solve(J(xc), -F(xc))
print(xc)

# Resposta da letra a: a jacobiana tá acima, e o sympy confirma (diferença ~0).
# Resposta das letras b e c: converge em 3 iterações, parando pelo critério ||F|| < 1e-8 (o do passo ainda tava em ~3e-5 na iteração 3).
#   q1 ≈ 0,07089 m³/s, q2 ≈ 0,04911 m³/s e h ≈ 6,030 (perda de carga comum às duas tubulações; a unidade segue a de K, provavelmente m se K estiver em s²/m⁵).
#   Nas tabelas dá pra ver a convergência quadrática: ||F|| vai 2,8e-1 -> 3,0e-4 -> 3,5e-10 (os dígitos corretos dobram a cada passo).
#   F1 é linear, então já era 0 em x0 e continua 0 em todas as iterações.
# Interpretação: a tubulação 1 (K = 1200, menos resistência) leva ≈ 59% da vazão e a 2 (K = 2500) ≈ 41%. Como a perda de carga h é a mesma
# nas duas (K1 q1² = K2 q2²), a razão das vazões é q1/q2 = sqrt(K2/K1) ≈ 1,443. As duas vazões deram positivas e somam 0,120, como tinha que ser.
