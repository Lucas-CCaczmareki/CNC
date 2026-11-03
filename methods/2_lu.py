# implementação do algoritmo de decomposição LU com pivotação parcial
# o objetivo é fatorar a matriz A em duas matrizes triangulares, L e U, pra depois resolver Ax = b
# resolvendo 2 sistemas triangulares (mais barato) em vez de refazer o escalonamento inteiro
# A x = b  ->  (L U) x = b  ->  L y = P b  e depois  U x = y

# A: matriz que a gente quer decompor (m linhas, n colunas)
# pivot: vetor que guarda as trocas de linha. linha i de PA = linha pivot[i] de A
# pdu: produto da diagonal de U (com o sinal trocado a cada troca de linha). Pra matriz quadrada isso é o det(A)
# info: informação sobre singularidade. 0 = matriz não singular. k = primeiro k onde u_kk = 0

# ps: o pseudocódigo é 1-based, python é 0-based. Só o info continua 1-based igual o livro

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

# main
# exemplo do slide da aula 8. Resposta esperada: x = (-3, 5, 0) e det(A) = -8
A = np.array([[3, 2, 4],
              [1, 1, 2],
              [4, 3, -2]])
b = np.array([1, 2, 3])

A_lu, pivot, pdu, info = lu(A)
L, U = extrair_lu(A_lu)
P = np.eye(len(pivot))[pivot] # monta a matriz P a partir do vetor pivot

print("L =\n", L)
print("U =\n", U)
print("pivot =", pivot)
print("det(A) =", pdu)
print("info =", info)

# confere se PA = LU. Se n for ~0 tem erro na montagem
print("||PA - LU|| =", np.linalg.norm(P @ A - L @ U))

x = resolver_lu(A_lu, pivot, b)
print("x =", x)
print("x (np.linalg.solve) =", np.linalg.solve(A, b)) # confere com a lib
