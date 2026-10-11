# relatório 2 - exercício 4: fluxo de potência de duas barras, F(V, δ) = 0 por Newton Modificado
# newton_modificado adaptado de methods/newton_modificado.py: a J é calculada e fatorada (LU) só em x(0), e reaproveitada em toda iteração
# o enunciado n dá o critério de parada, então usei o mesmo do exercício 3:
#   ||Δx||inf / max(||x(k+1)||inf, 1) < 1e-6  OU  ||F(x(k+1))||inf < 1e-8
# δ tá em radianos (as funções sin e cos do numpy esperam radiano)

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

V1 = 1.0   # pu
X = 0.5    # pu
PL = 0.6   # pu
QL = 0.2   # pu

# x = [V, δ]
def F(x):
    V, d = x
    return np.array([V1 * V / X * np.sin(d) - PL,
                     V1 * V / X * np.cos(d) - V**2 / X - QL])

# letra a) jacobiana analítica: J[i][j] = dFi/dxj, com x = (V, δ)
#   F1 = (V1 V / X) sin δ - PL           -> dF1/dV = (V1/X) sin δ           dF1/dδ =  (V1 V / X) cos δ
#   F2 = (V1 V / X) cos δ - V²/X - QL    -> dF2/dV = (V1/X) cos δ - 2V/X    dF2/dδ = -(V1 V / X) sin δ
def J(x):
    V, d = x
    return np.array([[V1 / X * np.sin(d),             V1 * V / X * np.cos(d)],
                     [V1 / X * np.cos(d) - 2 * V / X, -V1 * V / X * np.sin(d)]])

# confere a jacobiana da mão com a do sympy num ponto qualquer
sV, sd = sp.symbols('V d')
F_sym = sp.Matrix([V1 * sV / X * sp.sin(sd) - PL, V1 * sV / X * sp.cos(sd) - sV**2 / X - QL])
J_sym = sp.lambdify([sV, sd], F_sym.jacobian([sV, sd]), "numpy")
ponto = [0.8, 0.4]
print("a) J analítica no ponto", ponto, "=\n", J(ponto))
print("   |J analítica - J sympy| =", np.max(np.abs(J(ponto) - np.array(J_sym(*ponto)))))

def newton_modificado(F, J, x0, toler_rel, toler_res, k_max):
    x = np.array(x0, dtype=float)

    # a diferença pro newton: calcula J(x0) e fatora LU só UMA vez, antes do loop
    J_lu, pivot, pdu, info_lu = lu(J(x))
    if info_lu != 0:
        print("jacobiana singular em x0, escolha outro x0")
        return x, 0, 2, []

    k = 0
    info = 0
    historico = [[0, *x, None, None, np.max(np.abs(F(x)))]] # [k, x1..xn, dx, dr, ||F||]

    while True:
        # resolve J(x0) Δx = -F(x_k) reaproveitando a LU (só 2 substituições)
        dx = resolver_lu(J_lu, pivot, -F(x))
        x = x + dx
        k = k + 1
        norma_dx = np.max(np.abs(dx))
        dr = norma_dx / max(np.max(np.abs(x)), 1)
        norma_f = np.max(np.abs(F(x)))
        historico.append([k, *x, norma_dx, dr, norma_f])

        if dr < toler_rel or norma_f < toler_res:
            break
        if k >= k_max:
            info = 1
            break

    return x, k, info, historico

# main
# valor analítico das raízes pra conferir (com V1 = 1): V sin δ = PL X = a   e   V cos δ = V² + QL X
# elevando ao quadrado e somando: V² = a² + (V² + c)²  ->  u² + (2c - 1) u + (a² + c²) = 0, com u = V²
a = PL * X
c = QL * X
u = np.roots([1, 2 * c - 1, a**2 + c**2])
print("\nraízes analíticas: V =", np.sqrt(u), "| δ =", [np.arcsin(a / np.sqrt(ui)) for ui in u], "rad")

resultados = []
for x0 in ([1.0, 0.20], [0.45, 0.75]):
    print("\nb) Newton Modificado, x0 =", x0, "| det J(x0) =", np.linalg.det(J(x0)))
    x, k, info, historico = newton_modificado(F, J, x0, 1e-6, 1e-8, 500)
    imprimir_tabela(["k", "V", "δ (rad)", "||Δx||", "dr", "||F||"], historico)
    print(f"c) V = {x[0]:.6f} pu | δ = {x[1]:.6f} rad = {np.degrees(x[1]):.4f}° | iterações = {k} | info = {info}")
    resultados.append((x0, x, k))

print("\nd) comparação das duas raízes")
for x0, x, k in resultados:
    print(f"  x0 = {x0} -> V = {x[0]:.6f} pu, δ = {np.degrees(x[1]):.4f}°, {k} iterações")
print("  |V1 - V2| =", abs(resultados[0][1][0] - resultados[1][1][0]))

# Resposta da letra a: a jacobiana tá acima e o sympy confirma (diferença ~0).
# Resposta das letras b e c:
#   x0 = (1,0 ; 0,20) -> V ≈ 0,8031 pu e δ ≈ 0,3828 rad ≈ 21,9°
#   x0 = (0,45 ; 0,75) -> V ≈ 0,3938 pu e δ ≈ 0,8662 rad ≈ 49,6°
#   as duas batem com as raízes analíticas (V² = 0,6450 e 0,1550), e cada uma precisou de 18 e 11 iterações.
# Resposta da letra d: o sistema tem DUAS soluções (mesmo sistema, estimativas iniciais diferentes -> raízes diferentes), e cada chute cai
# na raiz da sua região. Em sistemas elétricos:
#   - a raiz de V ≈ 0,80 pu e δ ≈ 22° é a solução de alta tensão, a que faz sentido operar (barra de carga com tensão razoável).
#   - a raiz de V ≈ 0,39 pu e δ ≈ 50° é a solução de baixa tensão: matematicamente válida, mas tensão muito abaixo do aceitável (~0,95 a 1,05 pu)
#     e fisicamente instável/inviável. Aparece quando o chute inicial é ruim (V baixo e δ alto).
#   Então o x0 escolhe a raiz, e saber qual é a fisicamente útil exige conhecimento do problema, não só do método.
# Convergência: como a J é congelada, a convergência é linear: 18 iterações pro 1º chute e 11 pro 2º, com ||Δx|| caindo por um fator ~constante
# (≈ 0,5 e ≈ 0,3 por passo), bem diferente do Newton normal, que dobraria os dígitos corretos.
# Detalhe: det J tem sinal diferente nas duas raízes (≈ +1,57 na de alta tensão e ≈ -0,77 na de baixa). J é singular no ponto de máximo carregamento,
# que fica entre os dois ramos, e o sinal do det J separa o ramo estável do instável.
