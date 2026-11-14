# relatório 3 - exercício 4: força de arrasto, regressão linear e de potência
# usa as funções regress (methods/3_linear_regression.py) e metricas
# a linearização da letra b segue o caso C (potência) de methods/3_nonlinear_regression.py
# copiadas aqui pra esse arquivo rodar sozinho

import numpy as np
import matplotlib.pyplot as plt

def regress(x, y):
    # copia pra n estragar os vetores originais
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    sumx = 0.0
    sumy = 0.0
    sumxy = 0.0
    sumx2 = 0.0
    st = 0.0
    sr = 0.0

    # somatórios que aparecem nas equações normais
    for i in range(n):
        sumx = sumx + x[i]
        sumy = sumy + y[i]
        sumxy = sumxy + x[i] * y[i]
        sumx2 = sumx2 + x[i] * x[i]

    xm = sumx / n
    ym = sumy / n

    # se todos os x forem iguais, a reta é vertical e n dá pra escrever como y = a0 + a1 x
    den = n * sumx2 - sumx * sumx

    # compara os x direto, pq o den em float n dá 0 exato (ex: x = 0.7 sai ~1e-16)
    if np.all(x == x[0]):
        print("todos os x são iguais, n existe reta de regressão")
        return np.nan, np.nan, np.nan, np.nan, 1

    a1 = (n * sumxy - sumx * sumy) / den
    a0 = ym - a1 * xm

    # st = erro em volta da média, sr = erro em volta da reta
    for i in range(n):
        st = st + (y[i] - ym) ** 2
        sr = sr + (y[i] - a1 * x[i] - a0) ** 2

    # n - 2 pq a gente perdeu 2 graus de liberdade estimando a0 e a1
    if n > 2:
        syx = (sr / (n - 2)) ** 0.5
    else:
        syx = np.nan

    # st = 0 quando todos os y são iguais (reta horizontal), aí o r2 n faz sentido
    if st != 0:
        r2 = (st - sr) / st
    else:
        r2 = np.nan

    return a1, a0, syx, r2, 0

# Sr, RMSE e R² sempre na escala original dos dados, assim dá pra comparar modelos diferentes com a mesma régua
#   Sr   = soma de (y_i - modelo_i)²
#   RMSE = raiz(Sr / n)
#   R²   = 1 - Sr / St, com St = soma de (y_i - média de y)²
def metricas(y, y_modelo):
    y = np.array(y, dtype=float)
    y_modelo = np.array(y_modelo, dtype=float)
    sr = np.sum((y - y_modelo) ** 2)
    st = np.sum((y - np.mean(y)) ** 2)
    return sr, np.sqrt(sr / len(y)), 1 - sr / st

# main
v = [10, 20, 30, 40, 50, 60, 70, 80]                # m/s
F = [26, 94, 205, 360, 565, 790, 1090, 1420]        # N
v = np.array(v, dtype=float)
F = np.array(F, dtype=float)
n = len(v)

# ---------------------------------------------------------------- letra a) modelo linear, F = a0 + a1 v
a1, a0, syx, r2, info = regress(v, F)
F_lin = a0 + a1 * v
res_lin = F - F_lin
sr_lin, rmse_lin, r2_lin = metricas(F, F_lin)

print("a) modelo linear")
print(f"F(v) = {a0:.4f} + {a1:.4f} v | info = {info}")
print(f"Sr = {sr_lin:.4f} | RMSE = {rmse_lin:.4f} N | R² = {r2_lin:.6f}")
print(f"(a regress devolve syx = {syx:.4f} = raiz(Sr/(n-2)) e R² = {r2:.6f}, que bate com o meu)")
print("F(0) previsto =", a0, "N")
print("resíduos (N) =", np.round(res_lin, 2))
print("soma dos resíduos =", np.sum(res_lin))

# confere com a lib
print("\n(a1, a0) (np.polyfit) =", np.polyfit(v, F, 1))
# Resposta da letra a: a0 ≈ -325,786 N e a1 ≈ 19,8786 N·s/m, ou seja F(v) ≈ -325,79 + 19,879 v. Sr ≈ 81487,6 N², RMSE ≈ 100,93 N e R² ≈ 0,9532. (O syx da regress é raiz(Sr/(n-2)) ≈ 116,54 N: usa n-2 pq perdeu 2 graus de liberdade, e o RMSE usa n.)
#
# Só pelo R² = 0,95 o ajuste parece bom, mas n é: o intercepto dá F(0) = -325,8 N, uma força de arrasto negativa com o ar parado, o que n tem sentido físico. Os resíduos (gráfico da letra c) mostram o motivo.

# ---------------------------------------------------------------- letra b) modelo de potência, F = α v^β
# Caso C dos slides da aula 15 (methods/3_nonlinear_regression.py), função de potência y = b x^a, aqui F = α v^β (b = α, a = β)
# achata (transforma) a função de potência numa reta com ln (log natural): ln F = ln α + β ln v, reta em (ln v, ln F)
lv = np.log(v)
lF = np.log(F)
a1_ln, a0_ln, syx_log, r2_log, info = regress(lv, lF) # aplica a MMQ pra achar a reta na função achatada

# desfaz o achatamento: o a1 da reta é o β e o a0 é o ln α (por isso o exp)
beta = a1_ln
alpha = np.exp(a0_ln)

# Sr, RMSE e R² na escala original de F (a regress calcula tudo em ln, n dá pra comparar com o modelo linear)
F_pot = alpha * v ** beta
res_pot = F - F_pot
sr_pot, rmse_pot, r2_pot = metricas(F, F_pot)

print("b) modelo de potência")
print(f"F(v) = {alpha:.6f} v^{beta:.6f} | info = {info}")
print(f"Sr = {sr_pot:.4f} | RMSE = {rmse_pot:.4f} N | R² = {r2_pot:.6f}")
print(f"(R² da reta em ln: {r2_log:.6f}, outra escala, n é o R² que vale pro modelo)")
print("resíduos (N) =", np.round(res_pot, 2))
print("erro relativo (%) =", np.round(100 * res_pot / F, 2))

# confere com a lib
print("\n(β, ln α) (np.polyfit em ln) =", np.polyfit(lv, lF, 1))

# comparação dos dois modelos, na mesma régua
print("\nmodelo   |       Sr |   RMSE |       R²")
print(f"linear   | {sr_lin:>8.2f} | {rmse_lin:>6.2f} | {r2_lin:.6f}")
print(f"potência | {sr_pot:>8.2f} | {rmse_pot:>6.2f} | {r2_pot:.6f}")
print(f"Sr linear / Sr potência = {sr_lin / sr_pot:.1f}")
# Resposta da letra b: β ≈ 1,9278 e α ≈ 0,29805, ou seja F(v) ≈ 0,298 v^1,928. Sr ≈ 1252,06 N², RMSE ≈ 12,51 N e R² ≈ 0,99928.
#
# - Comparando na mesma escala (a de F): o Sr cai ~65 vezes em relação ao linear (81488 -> 1252) e o R² sobe de 0,953 pra 0,999.
# - O β ≈ 1,93 é praticamente 2: o arrasto aerodinâmico cresce com o quadrado da velocidade (F = ½ ρ Cd A v²), então o modelo de potência acerta a física e o linear n.
# - O R² da reta em ln (0,99977) é de outra escala, por isso recalculei tudo em F.
# - A linearização é o caso C (potência) dos slides da aula 15, o mesmo de methods/3_nonlinear_regression.py: ln em v e em F, regress na reta achatada, β = a1 e α = exp(a0).

# ---------------------------------------------------------------- letra c) gráficos
vs = np.linspace(10, 80, 200)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 14))

# gráfico 1: dados experimentais e os dois modelos
ax1.plot(v, F, "o", color="black", label="dados experimentais")
ax1.plot(vs, a0 + a1 * vs, label=f"linear: F = {a0:.2f} + {a1:.3f} v")
ax1.plot(vs, alpha * vs ** beta, label=f"potência: F = {alpha:.4f} v^{beta:.4f}")
ax1.set_xlabel("v (m/s)")
ax1.set_ylabel("F (N)")
ax1.set_title("Força de arrasto: dados e modelos")
ax1.legend()
ax1.grid(True)

# gráfico 2: resíduos do modelo linear (F_i - modelo(v_i))
ax2.axhline(0, color="gray")
ax2.plot(v, res_lin, "o--", color="tab:blue")
ax2.set_xlabel("v (m/s)")
ax2.set_ylabel("resíduo (N)")
ax2.set_title("Resíduos do modelo linear")
ax2.grid(True)

# gráfico 3: resíduos do modelo de potência
ax3.axhline(0, color="gray")
ax3.plot(v, res_pot, "o--", color="tab:orange")
ax3.set_xlabel("v (m/s)")
ax3.set_ylabel("resíduo (N)")
ax3.set_title("Resíduos do modelo de potência")
ax3.grid(True)

plt.tight_layout()
plt.show()
# Resposta da letra c:
#
# - Gráfico 1: a reta n acompanha a curvatura dos dados: fica abaixo dos pontos nas pontas (v = 10 e v = 80) e acima deles no meio. A curva de potência passa praticamente em cima de todos os pontos.
# - Gráfico 2 (linear): os resíduos têm formato de U: +153 N em v = 10, negativos no meio (até -109 N em v = 40) e +155,5 N em v = 80. Isso é erro sistemático, n aleatório: o modelo tem a forma errada (os dados são curvos). Um bom ajuste teria resíduos espalhados em volta de zero, sem padrão.
# - Gráfico 3 (potência): os resíduos são bem menores (entre -8,5 e +29,7 N) e sem o formato de U. Eles crescem em módulo com a velocidade, mas em erro relativo ficam entre ~0,6 % e ~2,9 % pra todos os pontos, pq o ajuste em ln minimiza erros relativos. Os 2 últimos resíduos são positivos e maiores (+15,2 e +29,7 N): o modelo subestima um pouco nas velocidades mais altas.
#
# Conclusão: o modelo de potência é o mais adequado pra esses dados (menor Sr e RMSE, R² maior, resíduos sem padrão e β ≈ 2 coerente com a física). Mesmo assim o R² do linear (0,95) parecia alto: só o gráfico de resíduos mostra que o modelo tá errado, então n dá pra julgar um ajuste só pelo R².
