# relatório 2 - exercício 1: transferência de calor, A T = b
# usa as funções lu / resolver_lu (methods/lu.py) e cholesky / resolver_cholesky (methods/cholesky.py)
# copiadas aqui pra esse arquivo rodar sozinho

import numpy as np

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

def cholesky(A):
    # copia pra n estragar a matriz original (o algoritmo escreve dentro da própria A)
    A = np.array(A, dtype=float)
    n = A.shape[0]

    det = 1.0
    info = 0

    # o cálculo avança coluna a coluna: 1o o elemento da diagonal, depois os que estão abaixo dele
    for j in range(n):
        # elemento da diagonal: g_jj = raiz(a_jj - soma dos g_jk²)
        soma = 0.0
        for k in range(j):
            soma = soma + A[j, k] * A[j, k]
        t = A[j, j] - soma # o que vai dentro da raiz

        if t > 0:
            A[j, j] = np.sqrt(t)
            r = 1 / A[j, j]
            det = det * t # t = g_jj², o produto de todos eles é o det(A)
        else:
            # radicando <= 0 em algum passo = A n é definida positiva. O próprio algoritmo é o teste
            info = j + 1
            print("matriz não é definida positiva")
            return A, det, info

        # elementos abaixo da diagonal (i > j): g_ij = (a_ij - soma de g_ik * g_jk) / g_jj
        for i in range(j + 1, n):
            soma = 0.0
            for k in range(j):
                soma = soma + A[i, k] * A[j, k]
            A[i, j] = (A[i, j] - soma) * r # multiplica por 1/g_jj (mesma coisa que dividir)

    # a G tá no triângulo inferior de A. O de cima ainda tem os valores originais, por isso o extrair_g
    return A, det, info

# pega só o triângulo inferior (diagonal inclusa) da matriz decomposta, pq acima da diagonal g_ij = 0
def extrair_g(A_g):
    return np.tril(A_g)

# resolve Ax = b usando a decomposição
def resolver_cholesky(A_g, b):
    G = extrair_g(A_g)
    n = G.shape[0]
    b = np.array(b, dtype=float)

    # G y = b  (substituição progressiva, de cima pra baixo)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(G[i, :i], y[:i])) / G[i, i]

    # G^T x = y  (substituição retroativa, de baixo pra cima)
    GT = G.T
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(GT[i, i+1:], x[i+1:])) / GT[i, i]

    return x

# imprime uma tabela simples: colunas = lista de nomes, linhas = lista de listas
def imprimir_tabela(colunas, linhas):
    print(" | ".join(f"{c:>12}" for c in colunas))
    print("-" * (15 * len(colunas)))
    for linha in linhas:
        print(" | ".join(f"{'-':>12}" if v is None else f"{v:>12.6g}" for v in linha))

# main
A = np.array([[ 7, -2, -1,  0],
              [-2,  6, -1, -1],
              [-1, -1,  7, -2],
              [ 0, -1, -2,  7]])
b = np.array([400, 40, 240, 80])
b_novo = np.array([480, 40, 240, 80])

# ---------------------------------------------------------------- letra a) simetria
# A é simétrica se A = A^T, ou seja a_ij = a_ji pra todo i, j
print("a) simétrica?", np.array_equal(A, A.T), "| max|A - A^T| =", np.max(np.abs(A - A.T)))
# Resposta da letra a: sim, A = A^T (a diferença é exatamente 0). Cholesky não é descartado por simetria.

# ---------------------------------------------------------------- letra b) dominância diagonal
# por linhas:  |a_ii| > soma dos |a_ij|, j != i
# por colunas: mesma coisa olhando pras colunas
print("\nb) dominância diagonal")
for i in range(4):
    soma_linha = np.sum(np.abs(A[i, :])) - abs(A[i, i])
    soma_coluna = np.sum(np.abs(A[:, i])) - abs(A[i, i])
    print(f"  linha {i+1}: {abs(A[i,i])} > {soma_linha} ? {abs(A[i,i]) > soma_linha} | coluna {i+1}: {abs(A[i,i])} > {soma_coluna} ? {abs(A[i,i]) > soma_coluna}")
# Resposta da letra b: A é estritamente diagonalmente dominante por linhas e por colunas (as 4 linhas e as 4 colunas passam).
# Isso garante que A é não singular e que Jacobi/Seidel convergiriam. Sendo simétrica com diagonal positiva e dominante, também é definida positiva.

# ---------------------------------------------------------------- letra c) autovalores / definida positiva
# A é simétrica, então uso eigvalsh (própria pra simétricas, devolve autovalores reais em ordem crescente)
autovalores = np.linalg.eigvalsh(A)
print("\nc) autovalores =", autovalores)
print("   todos > 0 ?", np.all(autovalores > 0))

# confirmação pelo critério de sylvester: todos os menores principais com det > 0
menores = [float(np.linalg.det(A[:k, :k])) for k in range(1, 5)]
print("   menores principais =", [round(m, 6) for m in menores])
print("   cond2(A) = λmax/λmin =", autovalores[-1] / autovalores[0])
# Resposta da letra c: autovalores ≈ 3,12 ; 6,23 ; 8,25 ; 9,41, todos positivos, e os menores principais (7, 38, 249, 1507) também.
# Logo A é simétrica e definida positiva: LU sem pivotação existe e é única, e Cholesky pode ser aplicado.
# O menor autovalor (3,12) longe de zero e cond2(A) = λmax/λmin ≈ 3 indicam que A é bem condicionada.

# ---------------------------------------------------------------- letra d) LU
A_lu, pivot, pdu, info = lu(A)
L, U = extrair_lu(A_lu)
T_lu = resolver_lu(A_lu, pivot, b)
print("\nd) LU")
print("L =\n", L)
print("U =\n", U)
print("pivot =", pivot, "| det(A) =", pdu, "| info =", info)
print("T (LU) =", T_lu)

# ---------------------------------------------------------------- letra e) Cholesky
A_g, det_ch, info_ch = cholesky(A)
G = extrair_g(A_g)
T_ch = resolver_cholesky(A_g, b)
print("\ne) Cholesky")
print("G =\n", G)
print("det(A) =", det_ch, "| info =", info_ch)
print("T (Cholesky) =", T_ch)
# os dois determinantes (LU e Cholesky) batem, e o menor principal de ordem 4 acima também (1507)

# ---------------------------------------------------------------- letra f) resíduo
res_lu = np.max(np.abs(b - A @ T_lu))
res_ch = np.max(np.abs(b - A @ T_ch))
print("\nf) ||b - A x||inf  LU =", res_lu, "| Cholesky =", res_ch)
print("   diferença entre as soluções =", np.max(np.abs(T_lu - T_ch)))
# Resposta da letra f: os dois resíduos ficam na ordem de 1e-14 (erro de arredondamento do ponto flutuante), ou seja,
# as duas soluções satisfazem o sistema até a precisão da máquina e são praticamente idênticas.

# ---------------------------------------------------------------- letra g) b novo
# aqui a gente aproveita a fatoração que já tá feita: só 2 substituições, n precisa fatorar de novo (esse é o ponto da aula 8)
T_novo = resolver_lu(A_lu, pivot, b_novo)
T_novo_ch = resolver_cholesky(A_g, b_novo)
delta = T_novo - T_lu
print("\ng) b novo")
print("T  =", T_lu)
print("T' =", T_novo, "(LU) |", T_novo_ch, "(Cholesky)")
print("ΔT =", delta)
print("ΔT em % =", 100 * delta / T_lu)
# Resposta da letra g: b1 subiu 80 (de 400 pra 480, +20%). As temperaturas sobem todas, e o efeito diminui com a "distância" do nó 1:
#   T1 sobe ≈ 13,4 °C (+16,6%), T2 ≈ 5,3 °C, T3 ≈ 3,1 °C e T4 ≈ 1,6 °C.
# Fisicamente b_i é a fonte de calor (ou condição de contorno) do nó i. Aquecer o nó 1 esquenta mais ele e os vizinhos diretos (T2 e T3, que
# têm a_12 e a_13 ≠ 0). O nó 4 n tem ligação direta com o 1 (a_14 = 0), então só sente o aumento "de segunda mão", via nós 2 e 3, e muito menos.
# O sistema é linear, então ΔT = A^-1 Δb: dobrar a fonte dobra o efeito. Todos os ΔT são positivos pq a inversa dessa A só tem entradas positivas
# (aumentar a fonte nunca esfria nenhum nó).
