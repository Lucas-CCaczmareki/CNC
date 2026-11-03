# implementação do método de Newton Modificado pra sistemas não lineares F(x) = 0 (esqueleto do ruggiero, cap 4)
# é o newton.py com UMA mudança: a jacobiana é calculada só uma vez, em x(0), e usada em todas as iterações
# J(x(0)) s = -F(x(k))  e depois  x(k+1) = x(k) + s
# como a matriz do sistema linear é sempre a mesma, fatora J(x(0)) = LU (com pivotação) uma única vez
# e cada iteração vira só 2 substituições (L y = P b e U s = y). É a economia da aula 8: fatorar uma vez, resolver muitas
# custo por iteração: n² (em vez de n³ do newton) e n derivadas parciais a menos por iteração

# o preço: a convergência deixa de ser quadrática e vira linear, então precisa de mais iterações
# funciona melhor quando o x0 já tá perto da raiz (se tiver longe, a jacobiana congelada é uma má aproximação)

# F: lista de expressões sympy [f1, f2, ..., fn]
# variaveis: lista de symbols do sympy na mesma ordem das colunas [x1, x2, ..., xn]
# x0: aproximação inicial (vetor)
# epsilon1: quão próximo de zero o F já tá. ||F(x)||inf < epsilon1
# epsilon2: distância entre a nova aproximação e a anterior. ||x(k+1) - x(k)||inf < epsilon2
# k_max: limite de iterações, protege contra divergência
# info: 0 = convergiu, 1 = n convergiu (parou por k_max), 2 = J(x0) singular (n dá pra fatorar)

# a norma usada é a do máximo (||v||inf = maior valor absoluto entre as componentes)
# as funções lu e resolver_lu são as do lu.py (copiadas aqui pra esse arquivo funcionar sozinho)

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


def newton_modificado(F, variaveis, x0, epsilon1, epsilon2, k_max):
    # usa a lib sympy pra montar a jacobiana (derivadas parciais) e converter F e J pra funções python
    F_expr = sp.Matrix(F)
    J_expr = F_expr.jacobian(variaveis)
    f_num = sp.lambdify(variaveis, F_expr, "numpy")
    j_num = sp.lambdify(variaveis, J_expr, "numpy")

    x = np.array(x0, dtype=float)

    # a diferença pro newton: calcula J(x0) e fatora LU só UMA vez, antes do loop
    J0 = np.array(j_num(*x), dtype=float)
    J_lu, pivot, pdu, info_lu = lu(J0)
    if info_lu != 0:
        print("jacobiana singular em x0, escolha outro x0")
        return x, 0, 2, []

    k = 0
    info = 0
    dx = None # ||x(k) - x(k-1)||, n existe no k = 0
    historico = [] # guarda os dados de cada iteração pra montar a tabela do relatório

    while True:
        # passo 1: calcula só F(x(k)). A J já tá fatorada
        Fx = np.array(f_num(*x), dtype=float).flatten()
        norma_f = np.max(np.abs(Fx))

        historico.append([k, *x, norma_f, dx])

        # passo 2: se F já tá próximo de zero, x(k) é a solução
        if norma_f < epsilon1:
            break

        # proteção contra divergência
        if k >= k_max:
            info = 1
            break

        # passo 3: resolve J(x0) s = -F(x(k)) reaproveitando a LU (só 2 substituições)
        s = resolver_lu(J_lu, pivot, -Fx)

        # passo 4: x(k+1) = x(k) + s
        x = x + s
        dx = np.max(np.abs(s))
        k = k + 1

        # passo 5: se o passo já é pequeno, convergiu (guarda a última linha antes de sair)
        if dx < epsilon2:
            Fx = np.array(f_num(*x), dtype=float).flatten()
            historico.append([k, *x, np.max(np.abs(Fx)), dx])
            break

        # passo 6: volta pro passo 1 com k + 1

    return x, k, info, historico

# main
# mesmo exemplo do newton.py (aula 12): x1 + x2 = 3 e x1² + x2² = 9, x0 = (1, 5), ε = 10^-4
# resposta esperada: x(1) = (-0,625 ; 3,625), x(2) = (-0,0586 ; 3,0586) -> converge pra (0, 3)
# o slide diz 8 iterações (só pelo critério do passo). Aqui pode dar menos pq testa o ||F|| também
x1, x2 = sp.symbols('x1 x2')
F = [x1 + x2 - 3,
     x1**2 + x2**2 - 9]

x, k, info, historico = newton_modificado(F, [x1, x2], [1, 5], 1e-4, 1e-4, 100)

print("k  x1  x2  ||F||  ||dx||")
for linha in historico:
    print(linha[0], [round(float(v), 7) if v is not None else None for v in linha[1:]])
print("x =", x)
print("iterações =", k)
print("info =", info)

# só pelo critério do passo (como no slide): ε1 = 0 desliga o teste do ||F||
x_s, k_s, info_s, _ = newton_modificado(F, [x1, x2], [1, 5], 0, 1e-4, 100)
print("\nsó pelo ||dx|| (como no slide): iterações =", k_s)

# exemplo 4 do slide: x0 = (0, 0) deixa a jacobiana singular, n dá nem pra fatorar
F4 = [x1**2 + x2**2 - 1,
      x1**2 - x2]
_, _, info_4, _ = newton_modificado(F4, [x1, x2], [0, 0], 1e-4, 1e-4, 100)
print("exemplo 4 (x0 = (0,0)): info =", info_4)
