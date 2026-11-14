# relatório 3 - exercício 1: sensor resistivo, polinômio interpolador de Lagrange
# usa as funções lagrange (methods/3_lagrange_interpolation.py)
# copiadas aqui pra esse arquivo rodar sozinho

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def lagrange(x, y, n, xi):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    soma = 0.0
    for i in range(n + 1):
        # começa com y_i e vai multiplicando pelos fatores de L_i(xi)
        produto = y[i]
        for j in range(n + 1):
            # o fator j = i é pulado (dava 0/0, e a definição do L_i já exclui ele)
            if i != j:
                produto = produto * (xi - x[j]) / (x[i] - x[j])
        soma = soma + produto

    return soma

# main
T = [0, 20, 40, 60, 80]                        # °C
R = [100.00, 107.79, 115.54, 123.24, 130.90]   # Ω
n = len(T) - 1                                 # grau do polinômio (5 pontos -> grau 4)

# ---------------------------------------------------------------- letra a) polinômio interpolador
# a função lagrange só avalia o polinômio num ponto, então pra ter a EXPRESSÃO de P4(T) monto a mesma soma com o sympy
# (em frações exatas, pra n perder nada no arredondamento)
# P_n(T) = soma de R_i * L_i(T), com L_i(T) = produto de (T - T_j)/(T_i - T_j), j != i
sT = sp.symbols('T')
P = 0
print("a) funções base de Lagrange")
for i in range(n + 1):
    Li = 1
    for j in range(n + 1):
        if i != j:
            Li = Li * (sT - T[j]) / sp.Integer(T[i] - T[j])
    print(f"  L_{i}(T) =", sp.factor(Li))
    P = P + sp.Rational(str(R[i])) * Li

P = sp.expand(P)
print("\nP4(T) =", P)
coef_P = [float(c) for c in sp.Poly(P, sT).all_coeffs()]
print("coeficientes (T^4 ... T^0) =", coef_P)

# confere com a lib (np.polyfit com grau n passando por todos os pontos dá o mesmo polinômio, pq ele é único)
print("\n|coef sympy - np.polyfit| =", np.max(np.abs(np.array(coef_P) - np.polyfit(T, R, n))))

# o polinômio tem que passar exatamente nos 5 pontos medidos
err_ex1 = max(abs(lagrange(T, R, n, Tk) - Rk) for Tk, Rk in zip(T, R))
print("passa nos pontos? max|P(T_i) - R_i| =", err_ex1)
# Resposta da letra a: cada L_i(T) vale 1 em T_i e 0 nos outros 4 pontos (as 5 funções base acima), e P4(T) = Σ R_i L_i(T) dá
#
# P4(T) = 100 + 0,390083 T - 1,4583e-5 T² - 8,3333e-7 T³ + 5,2083e-9 T⁴
#
# ou, em fração exata, T⁴/192000000 - T³/1200000 - 7T²/480000 + 4681T/12000 + 100. Bate com o np.polyfit (o polinômio de grau 4 que passa por 5 pontos é único) e passa nos 5 pontos com erro na ordem do arredondamento (~1e-14). Os termos de grau 3 e 4 são minúsculos: a curva é quase uma parábola, com uma leve concavidade pra baixo (o coeficiente de T² é negativo).

# ---------------------------------------------------------------- letra b) estimativas em T = 35 °C e T = 65 °C
print("b) estimativas")
T_est = [35, 65]
R_est = []
for Ti in T_est:
    Ri = lagrange(T, R, n, Ti)
    R_est.append(Ri)
    # conferência: o mesmo valor pelo polinômio do sympy, e uma interpolação linear só entre os 2 vizinhos (np.interp) pra ter noção de ordem de grandeza
    print(f"  R({Ti}) = {Ri:.4f} Ω | sympy = {float(P.subs(sT, Ti)):.4f} | linear entre vizinhos = {np.interp(Ti, T, R):.4f}")
# Resposta da letra b: R(35) ≈ 113,607 Ω e R(65) ≈ 125,158 Ω. O polinômio do sympy dá o mesmo valor, e a interpolação linear entre os dois vizinhos fica a menos de 0,01 Ω de distância (113,603 e 125,155): faz sentido, pq os dados quase não têm curvatura, então o P4 só corrige a reta nos centésimos.

# ---------------------------------------------------------------- letra c) gráfico
Ts = np.linspace(0, 80, 200)
Rs = [lagrange(T, R, n, t) for t in Ts] # a lagrange avalia 1 ponto por vez

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ts, Rs, label="P4(T) (Lagrange)")
ax.plot(T, R, "o", color="black", label="pontos medidos")
ax.plot(T_est, R_est, "s", color="red", label="estimativas")
for t, r in zip(T_est, R_est):
    ax.annotate(f"({t}; {r:.2f})", (t, r), textcoords="offset points", xytext=(8, -14))
ax.set_xlabel("T (°C)")
ax.set_ylabel("R (Ω)")
ax.set_title("Sensor resistivo: curva interpoladora")
ax.legend()
ax.grid(True)
plt.show()
# Resposta da letra c: a curva passa exatamente pelos 5 pontos pretos e os dois pontos estimados (vermelho) caem em cima dela, entre os pontos medidos. A resistência cresce de forma quase linear com a temperatura (~0,39 Ω/°C), com uma pequena concavidade. Como os pontos são igualmente espaçados e os dados são suaves, o grau 4 n oscila entre os nós.
