# relatório 3 - exercício 2: bomba centrífuga, interpolação de Newton em diferenças divididas
# usa as funções newton_interpolacao (methods/3_newton_interpolation.py)
# copiadas aqui pra esse arquivo rodar sozinho

import numpy as np
import matplotlib.pyplot as plt

def newton_interpolacao(x, y, n, xi):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # tabela das diferenças divididas. Só o triângulo superior esquerdo é usado (fdd[i, j] existe pra i + j <= n)
    fdd = np.zeros((n + 1, n + 1))

    # coluna 0 = os próprios valores de y (diferença dividida de ordem 0)
    for i in range(n + 1):
        fdd[i, 0] = y[i]

    # as outras colunas, j = ordem. Cada coluna usa a anterior, por isso o j é o laço de fora
    for j in range(1, n + 1):
        for i in range(n - j + 1):
            fdd[i, j] = (fdd[i + 1, j - 1] - fdd[i, j - 1]) / (x[i + j] - x[i])

    # monta o polinômio avaliando em xi, 1 termo por vez
    xterm = 1.0 # guarda o produto (xi - x0)(xi - x1)...(xi - x_{order-1}), reaproveita o do termo anterior
    yint = np.zeros(n + 1)
    ea = np.zeros(n)
    yint[0] = fdd[0, 0]

    for order in range(1, n + 1):
        xterm = xterm * (xi - x[order - 1])
        yint2 = yint[order - 1] + fdd[0, order] * xterm
        ea[order - 1] = yint2 - yint[order - 1] # erro estimado = o que esse termo acrescentou
        yint[order] = yint2

    return yint, ea, fdd

# imprime uma tabela simples: colunas = lista de nomes, linhas = lista de listas
def imprimir_tabela(colunas, linhas):
    print(" | ".join(f"{c:>12}" for c in colunas))
    print("-" * (15 * len(colunas)))
    for linha in linhas:
        print(" | ".join(f"{'-':>12}" if v is None else f"{v:>12.6g}" for v in linha))

# main
Q = [10, 18, 27, 39, 52]    # m³/h
eta = [58, 68, 76, 80, 77]  # %
n = len(Q) - 1              # grau do polinômio (5 pontos -> grau 4)

# ---------------------------------------------------------------- letra a) diferenças divididas de Newton
# a tabela só existe no triângulo i + j <= n, o resto da matriz fdd é zero de enchimento (vira "-" na impressão)
_, _, fdd = newton_interpolacao(Q, eta, n, Q[0])
linhas = [[Q[i], *[fdd[i, j] if i + j <= n else None for j in range(n + 1)]] for i in range(n + 1)]
print("a) tabela de diferenças divididas")
imprimir_tabela(["Q", "f[xi]", "ordem 1", "ordem 2", "ordem 3", "ordem 4"], linhas)
print("\ncoeficientes do polinômio (1a linha) =", fdd[0])

# confere o polinômio completo com a lib (np.polyfit com grau n passando por todos os pontos dá o mesmo polinômio, pq ele é único)
Qc = np.linspace(10, 52, 50)
dif = max(abs(newton_interpolacao(Q, eta, n, q)[0][n] - np.polyval(np.polyfit(Q, eta, n), q)) for q in Qc)
print("max|Newton - np.polyfit| em 50 pontos =", dif)

err_ex2 = max(abs(newton_interpolacao(Q, eta, n, Qk)[0][n] - ek) for Qk, ek in zip(Q, eta))
print("passa nos pontos? max|P(Q_i) - η_i| =", err_ex2)
# Resposta da letra a: a 1ª linha da tabela são os coeficientes, então
#
# P4(Q) = 58 + 1,25 (Q-10) - 0,0212418 (Q-10)(Q-18) - 1,79765e-4 (Q-10)(Q-18)(Q-27) + 7,00487e-6 (Q-10)(Q-18)(Q-27)(Q-39)
#
# Os coeficientes caem rápido de ordem pra ordem (1,25 -> 0,02 -> 1,8e-4 -> 7e-6), sinal de que os termos de grau mais alto são correções pequenas em cima dos anteriores. O polinômio é o mesmo do np.polyfit e passa nos 5 pontos.

# ---------------------------------------------------------------- letra b) estimativas em Q = 32 e Q = 45
# o enunciado escreve "estimar a vazão quando a eficiência for η(32) e η(45)", mas nos dados Q é a entrada e η é a saída.
# Então o que dá pra estimar é a EFICIÊNCIA nas vazões Q = 32 e Q = 45 m³/h (e é isso que o gráfico da letra c pede pra marcar)
Q_est = [32, 45]
eta_est = []
print("b) estimativas")
for Qi in Q_est:
    yint, ea, _ = newton_interpolacao(Q, eta, n, Qi)
    eta_est.append(yint[n])
    print(f"\nQ = {Qi} m³/h")
    imprimir_tabela(["grau", "η(Q) (%)", "ea"], [[k, yint[k], None if k == 0 else ea[k - 1]] for k in range(n + 1)])
# Resposta da letra b: η(32) ≈ 78,61 % e η(45) ≈ 79,33 % (grau 4). Nota: o enunciado pede "a vazão quando a eficiência for η(32)", mas pelos dados o que se estima é a eficiência nas vazões 32 e 45 m³/h, foi o que fiz.
#
# - Em Q = 32 a estimativa estabiliza: o termo do grau 4 acrescenta só -0,076 (os graus 3 e 4 dão 78,68 e 78,61).
# - Em Q = 45 a correção do último termo ainda é +0,71, ou seja, a estimativa é menos firme. É esperado: 45 fica entre 39 e 52, perto da ponta da tabela, e a curva muda de direção por ali (a eficiência sobe até 80 % e cai pra 77 %).

# ---------------------------------------------------------------- letra c) gráfico
Qs = np.linspace(10, 52, 200)
etas = [newton_interpolacao(Q, eta, n, q)[0][n] for q in Qs]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Qs, etas, label="P4(Q) (Newton)")
ax.plot(Q, eta, "o", color="black", label="pontos medidos")
ax.plot(Q_est, eta_est, "s", color="red", label="estimativas")
for q, e in zip(Q_est, eta_est):
    ax.annotate(f"({q}; {e:.2f})", (q, e), textcoords="offset points", xytext=(8, -16))
ax.set_xlabel("Q (m³/h)")
ax.set_ylabel("η (%)")
ax.set_title("Bomba centrífuga: curva interpoladora")
ax.legend()
ax.grid(True)
plt.show()

# máximo da curva interpolada (aproxima o ponto de melhor eficiência da bomba)
i_max = np.argmax(etas)
print(f"máximo de P4 em 10 <= Q <= 52: η = {etas[i_max]:.2f} % em Q = {Qs[i_max]:.1f} m³/h")
# Resposta da letra c: a curva passa pelos 5 pontos pretos e os dois pontos estimados (vermelho) ficam sobre ela. A eficiência sobe rápido com a vazão no começo e depois achata, com um máximo de ~80,0 % em Q ≈ 39,5 m³/h, praticamente em cima do ponto medido de 39 m³/h (80 %). Depois disso a eficiência cai (77 % em 52 m³/h). É um comportamento típico de curva de bomba: tem uma vazão de melhor eficiência, e longe dela a eficiência cai. Esse máximo vem do polinômio, n foi medido, então vale só como estimativa.

# ---------------------------------------------------------------- letra d) novo ponto Q = 60 m³/h e η = 72 %
# o ponto novo vai no FIM das listas. O Newton n exige x ordenado, e assim as linhas antigas da tabela ficam como estavam
Q_novo = Q + [60]
eta_novo = eta + [72]
n_novo = len(Q_novo) - 1

_, _, fdd_novo = newton_interpolacao(Q_novo, eta_novo, n_novo, Q_novo[0])
linhas = [[Q_novo[i], *[fdd_novo[i, j] if i + j <= n_novo else None for j in range(n_novo + 1)]] for i in range(n_novo + 1)]
print("d) tabela com o ponto novo")
imprimir_tabela(["Q", "f[xi]", "ordem 1", "ordem 2", "ordem 3", "ordem 4", "ordem 5"], linhas)

# os coeficientes a0...a4 (1a linha) n mudaram. O único termo novo é a5 = f[x0,...,x5]
print("\ncoeficientes a0...a4 iguais aos de antes?", np.allclose(fdd_novo[0, :n + 1], fdd[0]))
a5 = fdd_novo[0, n_novo]
print("a5 = f[x0,...,x5] =", a5)

# a diagonal nova (1 entrada por coluna) é a única parte da tabela que foi calculada de novo
print("diagonal nova =", [float(fdd_novo[n_novo - j, j]) for j in range(n_novo + 1)])

# conferência: a5 é o que falta pro P5 passar no ponto novo.  a5 = (72 - P4(60)) / produto de (60 - x_i)
p4_60 = newton_interpolacao(Q, eta, n, 60)[0][n]
prod = np.prod([60 - q for q in Q])
print(f"P4(60) = {p4_60:.4f} (extrapolação, o ponto real é 72) | (72 - P4(60)) / prod = {(72 - p4_60) / prod}")

# efeito nas estimativas: P5 - P4 é só o termo novo, a5 (Q - x0)...(Q - x4)
print("\nefeito nas estimativas")
for Qi in Q_est:
    y4 = newton_interpolacao(Q, eta, n, Qi)[0][n]
    y5 = newton_interpolacao(Q_novo, eta_novo, n_novo, Qi)[0][n_novo]
    print(f"  Q = {Qi}: P4 = {y4:.4f} | P5 = {y5:.4f} | diferença = {y5 - y4:.4f} | a5 * prod = {a5 * np.prod([Qi - q for q in Q]):.4f}")

# gráfico extra: P4 e P5 com o ponto novo
Qs2 = np.linspace(10, 60, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Qs2, [newton_interpolacao(Q, eta, n, q)[0][n] for q in Qs2], "--", label="P4(Q)")
ax.plot(Qs2, [newton_interpolacao(Q_novo, eta_novo, n_novo, q)[0][n_novo] for q in Qs2], label="P5(Q)")
ax.plot(Q, eta, "o", color="black", label="pontos medidos")
ax.plot([60], [72], "s", color="red", label="ponto novo (60 ; 72)")
ax.set_xlabel("Q (m³/h)")
ax.set_ylabel("η (%)")
ax.set_title("Bomba centrífuga: P4 x P5")
ax.legend()
ax.grid(True)
plt.show()
# Resposta da letra d: o termo novo é a5 = f[x0,...,x5] ≈ -1,3988e-7, e P5(Q) = P4(Q) + a5 (Q-10)(Q-18)(Q-27)(Q-39)(Q-52).
#
# - Só a diagonal nova da tabela foi calculada (a última linha, com os valores 72, -0,625, -0,01877, 1,149e-4, 1,06e-8 e -1,3988e-7). Todo o resto, inclusive os coeficientes a0...a4, ficou igual.
# - O a5 é exatamente o que falta pro polinômio passar no ponto novo: P4 extrapolaria P4(60) ≈ 73,63 e o ponto real é 72, então o termo novo tem que cobrir essa diferença de -1,63. A conta a5 = (72 - P4(60)) / Π(60 - x_i) reproduz o valor da tabela (conferência acima).
# - O efeito nas estimativas é pequeno: η(32) vai de 78,605 pra 78,575 e η(45) de 79,334 pra 79,434. Dá pra ver no gráfico que P4 e P5 quase se sobrepõem dentro de [10, 52] e só se separam perto do ponto novo.
#
# Por que a forma de Newton é conveniente quando chegam dados novos: cada termo do polinômio só depende dos pontos anteriores, então um ponto novo vira uma linha a mais na tabela e um termo a mais no polinômio, sem mexer em nada do que já foi calculado (custo ~n operações). No Lagrange todas as funções base L_i mudam quando entra um ponto (refaz tudo), e na forma monomial a0 + a1 x + ... seria preciso resolver o sistema de Vandermonde de novo. De quebra, o próprio termo novo serve de estimativa do erro (ea): se ele é pequeno, o grau anterior já tava bom.
