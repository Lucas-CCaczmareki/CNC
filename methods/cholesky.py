# implementação do algoritmo de decomposição de Cholesky (fig 2.10 do livro)
# é uma simplificação da LU pra quando a matriz A é simétrica e definida positiva
# nesse caso a segunda matriz é só a transposta da primeira: A = G G^T
# guarda metade dos elementos e faz metade das operações da LU, e n precisa de pivotação
# A x = b  ->  (G G^T) x = b  ->  G y = b  e depois  G^T x = y

# A: matriz quadrada simétrica e definida positiva (n x n)
# det: determinante de A, sai de graça da fatoração
# info: informação sobre A. 0 = matriz definida positiva. k = primeiro k onde g_kk n é positivo

# ps: o pseudocódigo do livro é 1-based, python é 0-based. Só o info continua 1-based igual o livro
# ps2: a matriz G é a L do livro. Usei G pq é como a professora chama nos slides

import numpy as np

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

# main
# exemplo do slide da aula 9. Resposta esperada: x = (1, 1, 2) e det(A) = 2
A = np.array([[1, 1, 0],
              [1, 2, -1],
              [0, -1, 3]])
b = np.array([2, 1, 5])

A_g, det, info = cholesky(A)
G = extrair_g(A_g)

print("G =\n", G)
print("det(A) =", det)
print("info =", info)

# confere se G G^T = A. Se n for ~0 tem erro
print("||A - G G^T|| =", np.linalg.norm(A - G @ G.T))

x = resolver_cholesky(A_g, b)
print("x =", x)
print("x (np.linalg.solve) =", np.linalg.solve(A, b)) # confere com a lib
print("G (np.linalg.cholesky) =\n", np.linalg.cholesky(A))

# exercício 2 da aula 9: simétrica mas n é definida positiva. O algoritmo tem que falhar no passo 2
print("\nteste matriz n definida positiva:")
_, _, info2 = cholesky(np.array([[1, 2], [2, 1]]))
print("info =", info2)
