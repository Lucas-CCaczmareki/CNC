# implementação do algoritmo de Gauss-Jacobi (fig 2.11 do livro)
# é um método iterativo estacionário: parte de uma aproximação inicial x(0) e gera uma sequência x(k)
# que (se convergir) chega na solução de Ax = b. Em vez de escalonar a matriz, só repete uma conta simples
# x_i(k+1) = (b_i - soma dos a_ij * x_j(k), com j != i) / a_ii

# a diferença pro Seidel: no Jacobi TODAS as componentes do x novo usam só valores da iteração anterior (k)
# o vetor novo só substitui o antigo quando as n contas estão prontas. Por isso ele é paralelizável

# A: matriz dos coeficientes (n x n), os a_ii tem que ser != 0
# b: vetor dos termos independentes
# toler: tolerância (epsilon) do critério de parada
# iter_max: limite de iterações, protege contra divergência
# x: vetor solução (ou o último vetor calculado, se n convergiu)
# k: número de iterações feitas
# info: 0 = convergiu, 1 = n convergiu (parou por iter_max)

# o critério de parada é o erro relativo entre iterações (dr dos slides):
# dr = ||x(k) - x(k-1)||inf / ||x(k)||inf   (norma do máximo = maior valor absoluto das componentes)

# convergência garantida se a matriz for diagonalmente dominante por linhas (ou colunas), ou se ρ(M) < 1
# mas se n for, n dá pra concluir nada: pode convergir mesmo assim

import numpy as np

def jacobi(A, b, toler, iter_max):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # valor inicial de x: x(0) = b / diagonal de A (é o que o algoritmo do livro faz)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] / A[i, i]

    k = 0
    historico = [] # guarda os dados de cada iteração pra montar a tabela do relatório

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

        # norma do máximo da diferença (numerador) e do vetor novo (denominador)
        norma_num = 0.0
        norma_den = 0.0
        for i in range(n):
            t = abs(v[i] - x[i])
            if t > norma_num:
                norma_num = t
            if abs(v[i]) > norma_den:
                norma_den = abs(v[i])
            x[i] = v[i] # agora sim troca o vetor antigo pelo novo

        norma_rel = norma_num / norma_den
        historico.append([k, *x, norma_rel])

        # teste de convergência: ou chegou na tolerância, ou estourou o limite de iterações
        if norma_rel <= toler or k >= iter_max:
            break

    if norma_rel <= toler:
        info = 0
    else:
        info = 1

    return x, k, info, historico

# main
# exemplo do slide da aula 11, ε = 0,05. Solução exata: (1, -2, 1)
# resposta esperada: 3 iterações, x ≈ (0,9994 ; -1,9888 ; 0,9984) e dr = 0,1828 -> 0,0606 -> 0,0163
A = np.array([[10, 2, 1],
              [1, 5, 1],
              [2, 3, 10]])
b = np.array([7, -8, 6])

x, k, info, historico = jacobi(A, b, 0.05, 100)

print("k  x1  x2  x3  dr")
for linha in historico:
    print(linha[0], [round(float(v), 4) for v in linha[1:]])
print("x =", x)
print("iterações =", k)
print("info =", info)

# exercício 4 da aula 11: n é diagonalmente dominante, o Jacobi tem que divergir (info = 1)
x2, k2, info2, _ = jacobi([[1, 3], [2, 1]], [4, 3], 0.01, 10)
print("\nteste sistema que diverge:")
print("x =", x2, "| iterações =", k2, "| info =", info2)

# trocando as linhas fica diagonalmente dominante e converge pra (1, 1)
x3, k3, info3, _ = jacobi([[2, 1], [1, 3]], [3, 4], 0.01, 10)
print("\ntrocando as linhas:")
print("x =", x3, "| iterações =", k3, "| info =", info3)
