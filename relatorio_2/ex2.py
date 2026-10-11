# relatório 2 - exercício 2: rede de correntes, A V = b, com Gauss-Jacobi e Gauss-Seidel
# as funções são as de methods/jacobi.py e methods/seidel.py com 2 adaptações que o enunciado pede:
#   1) x(0) é parâmetro (aqui x(0) = 0). Nos methods o x(0) era b/diagonal
#   2) critério de parada duplo e SIMULTÂNEO (os dois precisam valer ao mesmo tempo):
#        ||x(k+1) - x(k)||inf / max(||x(k+1)||inf, 1) < 1e-6   e   ||b - A x(k+1)||inf < 1e-8
#      (nos methods o dr dividia só por ||x(k+1)||inf)

import numpy as np

# toler_rel: tolerância do erro relativo entre iterações
# toler_res: tolerância do resíduo ||b - A x||inf
def jacobi(A, b, x0, toler_rel, toler_res, iter_max):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.array(x0, dtype=float)

    k = 0
    historico = [] # [k, x1..xn, dr, resíduo]

    while True:
        k = k + 1

        # v é o vetor novo. A gente n mexe no x enquanto calcula, senão virava Seidel
        v = np.zeros(n)
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if i != j:
                    soma = soma + A[i, j] * x[j]
            v[i] = (b[i] - soma) / A[i, i]

        dr = np.max(np.abs(v - x)) / max(np.max(np.abs(v)), 1)
        x = v
        res = np.max(np.abs(b - A @ x))
        historico.append([k, *x, dr, res])

        # os 2 critérios têm que valer juntos
        if (dr < toler_rel and res < toler_res) or k >= iter_max:
            break

    info = 0 if (dr < toler_rel and res < toler_res) else 1
    return x, k, info, historico

def seidel(A, b, x0, toler_rel, toler_res, iter_max):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.array(x0, dtype=float)

    k = 0
    historico = []

    while True:
        k = k + 1
        x_ant = x.copy() # guarda o x(k) pra calcular a diferença depois

        for i in range(n):
            # aqui o x já tem os valores novos pra j < i e os antigos pra j > i
            soma = 0.0
            for j in range(n):
                if i != j:
                    soma = soma + A[i, j] * x[j]
            x[i] = (b[i] - soma) / A[i, i] # atualiza na hora

        dr = np.max(np.abs(x - x_ant)) / max(np.max(np.abs(x)), 1)
        res = np.max(np.abs(b - A @ x))
        historico.append([k, *x, dr, res])

        if (dr < toler_rel and res < toler_res) or k >= iter_max:
            break

    info = 0 if (dr < toler_rel and res < toler_res) else 1
    return x, k, info, historico

# imprime uma tabela simples: colunas = lista de nomes, linhas = lista de listas
def imprimir_tabela(colunas, linhas):
    print(" | ".join(f"{c:>12}" for c in colunas))
    print("-" * (15 * len(colunas)))
    for linha in linhas:
        print(" | ".join(f"{'-':>12}" if v is None else f"{v:>12.6g}" for v in linha))

# main
A = np.array([[10, -2, -1,  0,  0],
              [-2,  9, -2, -1,  0],
              [-1, -2, 10, -2, -1],
              [ 0, -1, -2,  9, -2],
              [ 0,  0, -1, -2,  8]])
b = np.array([12, 5, 0, 5, 10])
x0 = np.zeros(5)
n = 5

# ---------------------------------------------------------------- letra a) verificações prévias
print("a) dominância diagonal")
for i in range(n):
    soma_linha = np.sum(np.abs(A[i, :])) - abs(A[i, i])
    soma_coluna = np.sum(np.abs(A[:, i])) - abs(A[i, i])
    print(f"  linha {i+1}: {abs(A[i,i])} > {soma_linha} ? {abs(A[i,i]) > soma_linha} | coluna {i+1}: {abs(A[i,i])} > {soma_coluna} ? {abs(A[i,i]) > soma_coluna}")

# critério de sassenfeld (só vale pro seidel): beta_i = (soma_{j<i} |a_ij| beta_j + soma_{j>i} |a_ij|) / |a_ii|
beta = np.zeros(n)
for i in range(n):
    s = 0.0
    for j in range(n):
        if j < i:
            s = s + abs(A[i, j]) * beta[j]
        elif j > i:
            s = s + abs(A[i, j])
    beta[i] = s / abs(A[i, i])
print("  sassenfeld: beta =", np.round(beta, 4), "| max =", round(float(np.max(beta)), 4))

# raio espectral das matrizes de iteração (condição necessária e suficiente). M_jacobi = -D^-1 (L+U), M_seidel = -(D+L)^-1 U
D = np.diag(np.diag(A.astype(float)))
M_j = -np.linalg.inv(D) @ (A - D)
M_s = -np.linalg.inv(np.tril(A.astype(float))) @ np.triu(A.astype(float), 1)
print("  rho(M_jacobi) =", max(abs(np.linalg.eigvals(M_j))), "| rho(M_seidel) =", max(abs(np.linalg.eigvals(M_s))))
# Resposta da letra a: A é estritamente diagonalmente dominante por linhas e por colunas (10>3, 9>5, 10>6, 9>5, 8>3 nas linhas).
# Sassenfeld: max beta ≈ 0,41 < 1. Raio espectral: ρ(M_J) ≈ 0,50 < 1 e ρ(M_S) ≈ 0,26 < 1.
# Os 3 testes garantem convergência dos dois métodos. E o ρ já prevê que o Seidel é mais rápido: ρ(M_S) ≈ ρ(M_J)² (0,26 ≈ 0,25, comum em matrizes simétricas como essa).

# ---------------------------------------------------------------- letra b) Gauss-Jacobi
x_j, k_j, info_j, hist_j = jacobi(A, b, x0, 1e-6, 1e-8, 1000)
print("\nb) Gauss-Jacobi")
imprimir_tabela(["k", "V1", "V2", "V3", "V4", "V5", "dr", "||b-Ax||"], hist_j)
print("V =", x_j, "| iterações =", k_j, "| info =", info_j)

# ---------------------------------------------------------------- letra c) Gauss-Seidel
x_s, k_s, info_s, hist_s = seidel(A, b, x0, 1e-6, 1e-8, 1000)
print("\nc) Gauss-Seidel")
imprimir_tabela(["k", "V1", "V2", "V3", "V4", "V5", "dr", "||b-Ax||"], hist_s)
print("V =", x_s, "| iterações =", k_s, "| info =", info_s)

# ---------------------------------------------------------------- letra d) critério de parada
# já tá dentro das funções (toler_rel = 1e-6 e toler_res = 1e-8, simultâneos). As colunas dr e ||b-Ax|| das tabelas mostram os dois em cada iteração.
# o resíduo (1e-8) é o critério que manda (Jacobi para na it. 30 e Seidel na 17): o dr chega em 1e-6 antes. Por isso o "simultaneamente" importa: parar só pelo dr (Jacobi na it. 20, Seidel na it. 12) deixaria resíduo de ~1e-6 a 1e-5, bem acima de 1e-8.

# ---------------------------------------------------------------- letra e) comparação
x_ref = np.linalg.solve(A, b) # referência direta (LU do numpy)
print("\ne) comparação")
print("referência (np.linalg.solve) =", x_ref)
print(f"Jacobi : {k_j} iterações | erro vs ref = {np.max(np.abs(x_j - x_ref)):.3e} | resíduo = {np.max(np.abs(b - A @ x_j)):.3e}")
print(f"Seidel : {k_s} iterações | erro vs ref = {np.max(np.abs(x_s - x_ref)):.3e} | resíduo = {np.max(np.abs(b - A @ x_s)):.3e}")
print("diferença Jacobi - Seidel =", np.max(np.abs(x_j - x_s)))
# Resposta da letra e: as duas soluções coincidem com a direta até ~1e-9: V ≈ (1,523 ; 1,211 ; 0,808 ; 1,239 ; 1,661).
# O Seidel chega lá em 17 iterações contra 30 do Jacobi (~43% a menos, na linha do que o ρ previa), usando só 1 vetor.
# O Jacobi só ganha se der pra paralelizar as n contas de cada iteração.
# Interpretação: as tensões nodais são todas positivas e na mesma ordem de grandeza. V5 é a maior (b5 = 10 e a menor diagonal, 8) e
# V3 a menor (b3 = 0: o nó 3 n tem fonte, só recebe tensão dos vizinhos).
