# (versão do ruggiero, cap 4. Pra versão do burden ver newton_burden.py)
# implementação do método de Newton pra sistemas não lineares F(x) = 0 (ruggiero, cap 4)
# é o mesmo newton-raphson de 1 variável, só que agora x é um vetor e a derivada vira a matriz jacobiana J(x)
# a ideia: linearizar F em volta de x(k) e resolver o sistema linear que sobra
# F(x(k)) + J(x(k)) (x - x(k)) = 0  ->  J(x(k)) s = -F(x(k))  e depois  x(k+1) = x(k) + s
# ou seja, cada iteração de newton custa a resolução de um sistema linear completo

# F: lista de expressões sympy [f1, f2, ..., fn]
# variaveis: lista de symbols do sympy na mesma ordem das colunas [x1, x2, ..., xn]
# x0: aproximação inicial (vetor)
# epsilon1: quão próximo de zero o F já tá. ||F(x)||inf < epsilon1
# epsilon2: distância entre a nova aproximação e a anterior. ||x(k+1) - x(k)||inf < epsilon2
# k_max: limite de iterações, protege contra divergência
# info: 0 = convergiu, 1 = n convergiu (parou por k_max), 2 = jacobiana singular (n dá pra resolver J s = -F)

# a norma usada é a do máximo (||v||inf = maior valor absoluto entre as componentes)
# o sistema J s = -F é resolvido com a nossa fatoração LU (lu e resolver_lu do lu.py, copiadas aqui pra esse arquivo rodar sozinho)
# como a J muda a cada iteração, a LU é refeita a cada iteração (diferente do newton modificado, que fatora uma vez só)
# ps: a escolha do x0 importa. Com x0 diferente o método pode ir pra outra raiz, ou nem convergir

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

def newton(F, variaveis, x0, epsilon1, epsilon2, k_max):
    # usa a lib sympy pra montar a jacobiana (derivadas parciais) e converter F e J pra funções python
    # (mesma ideia do newtonraphson.py, só que com matriz)
    F_expr = sp.Matrix(F)
    J_expr = F_expr.jacobian(variaveis)
    f_num = sp.lambdify(variaveis, F_expr, "numpy")
    j_num = sp.lambdify(variaveis, J_expr, "numpy")

    x = np.array(x0, dtype=float)
    k = 0
    info = 0
    dx = None # ||x(k) - x(k-1)||, n existe no k = 0
    historico = [] # guarda os dados de cada iteração pra montar a tabela do relatório

    while True:
        # passo 1: calcula F(x(k)) e J(x(k))
        Fx = np.array(f_num(*x), dtype=float).flatten()
        Jx = np.array(j_num(*x), dtype=float)
        norma_f = np.max(np.abs(Fx))

        historico.append([k, *x, norma_f, dx])

        # passo 2: se F já tá próximo de zero, x(k) é a solução
        if norma_f < epsilon1:
            break

        # proteção contra divergência
        if k >= k_max:
            info = 1
            break

        # passo 3: resolve J s = -F. Fatora a J(x(k)) em LU (de novo, pq a J mudou) e resolve com 2 substituições
        # Se J for singular n tem solução (ex: ponto inicial ruim), e o info da lu avisa
        J_lu, pivot, pdu, info_lu = lu(Jx)
        if info_lu != 0:
            print("jacobiana singular, escolha outro x0")
            info = 2
            break
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
# exemplo do slide da aula 12: interseção de uma reta com uma circunferência de raio 3
# f1 = x1 + x2 - 3 = 0
# f2 = x1² + x2² - 9 = 0
# x0 = (1, 5), ε = 10^-4. Duas soluções: (0, 3) e (3, 0)
# resposta esperada: x(1) = (-0,625 ; 3,625), x(2) = (-0,0919 ; 3,0919) -> converge pra (0, 3)
x1, x2 = sp.symbols('x1 x2')
F = [x1 + x2 - 3,
     x1**2 + x2**2 - 9]

x, k, info, historico = newton(F, [x1, x2], [1, 5], 1e-4, 1e-4, 100)

print("k  x1  x2  ||F||  ||dx||")
for linha in historico:
    print(linha[0], [round(float(v), 7) if v is not None else None for v in linha[1:]])
print("x =", x)
print("iterações =", k)
print("info =", info)

# exemplo 2 do slide: equação transcendente, x0 = (1,5 ; 2). Solução exata: (1, 1)
F2 = [x1**2 + x2**2 - 2,
      sp.exp(x1 - 1) + x2**3 - 2]
x_2, k_2, info_2, _ = newton(F2, [x1, x2], [1.5, 2], 1e-4, 1e-4, 100)
print("\nexemplo 2: x =", x_2, "| iterações =", k_2, "| info =", info_2)

# exemplo 4 do slide: x0 = (0, 0) deixa a jacobiana singular, o método tem que parar na 1a iteração
F4 = [x1**2 + x2**2 - 1,
      x1**2 - x2]
x_4, k_4, info_4, _ = newton(F4, [x1, x2], [0, 0], 1e-4, 1e-4, 100)
print("\nexemplo 4 (x0 = (0,0)): info =", info_4)
x_4, k_4, info_4, _ = newton(F4, [x1, x2], [1, 1], 1e-4, 1e-4, 100)
print("exemplo 4 (x0 = (1,1)): x =", x_4, "| iterações =", k_4, "| info =", info_4)
