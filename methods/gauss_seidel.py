# implementação do algoritmo de Gauss-Seidel (fig 2.12 do livro)
# é igual o Jacobi, só que usa o valor MAIS RECENTE de cada componente assim que ele é calculado
# no Jacobi o x novo só substitui o antigo no final da iteração. Aqui o x(i) é atualizado na hora,
# então o x(i+1) já usa o x(i) novo, o x(i+2) usa os 2 novos, e assim por diante (encadeamento)
# x_i(k+1) = (b_i - soma dos a_ij * x_j, com j != i) / a_ii   onde os x_j com j < i já são da iteração k+1

# ganha velocidade (em geral converge ~2x mais rápido) e economiza memória (1 vetor em vez de 2)
# mas perde o paralelismo do Jacobi, pq cada conta depende da anterior

# A: matriz dos coeficientes (n x n), os a_ii tem que ser != 0
# b: vetor dos termos independentes
# toler: tolerância (epsilon) do critério de parada
# iter_max: limite de iterações, protege contra divergência
# x: vetor solução (ou o último vetor calculado, se n convergiu)
# k: número de iterações feitas
# info: 0 = convergiu, 1 = n convergiu (parou por iter_max)

# o critério de parada é o mesmo do Jacobi, erro relativo entre iterações (dr dos slides):
# dr = ||x(k) - x(k-1)||inf / ||x(k)||inf

# convergência garantida se a matriz for diagonalmente dominante por linhas, ou pelo critério de Sassenfeld (max β < 1)
# se n for, n dá pra concluir nada: existem sistemas que o Jacobi converge e o Seidel n (e o contrário)

import numpy as np

def seidel(A, b, toler, iter_max):
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
        norma_num = 0.0
        norma_den = 0.0

        for i in range(n):
            # aqui o x já tem os valores novos pra j < i e os antigos pra j > i. Essa é a diferença pro Jacobi
            soma = 0.0
            for j in range(n):
                if i != j:
                    soma = soma + A[i, j] * x[j]

            v = x[i]                        # guarda o valor antigo pra calcular a diferença
            x[i] = (b[i] - soma) / A[i, i]  # atualiza na hora

            t = abs(v - x[i])
            if t > norma_num:
                norma_num = t
            if abs(x[i]) > norma_den:
                norma_den = abs(x[i])

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
# mesmo exemplo do jacobi (slide da aula 11), ε = 0,05. Solução exata: (1, -2, 1)
# resposta esperada: 2 iterações, x ≈ (0,98424 ; -1,993168 ; 1,0011024) e dr = 0,1996 -> 0,0407
A = np.array([[10, 2, 1],
              [1, 5, 1],
              [2, 3, 10]])
b = np.array([7, -8, 6])

x, k, info, historico = seidel(A, b, 0.05, 100)

print("k  x1  x2  x3  dr")
for linha in historico:
    print(linha[0], [round(float(v), 6) for v in linha[1:]])
print("x =", x)
print("iterações =", k)
print("info =", info)

# exercício 3 da aula 11: sistema do exercício 1, ε = 0,01. Compara o número de iterações com o jacobi
A2 = [[5, 1, 1], [3, 4, 1], [3, 3, 6]]
b2 = [5, 6, 0]
x2, k2, info2, historico2 = seidel(A2, b2, 0.01, 100)
print("\nexercício 3 (seidel):")
for linha in historico2:
    print(linha[0], [round(float(v), 6) for v in linha[1:]])
print("x =", x2, "| iterações =", k2, "| info =", info2)
print("x (np.linalg.solve) =", np.linalg.solve(A2, b2)) # confere com a lib
