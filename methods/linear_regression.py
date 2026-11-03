## implementação do algoritmo de regressão linear por mínimos quadrados (fig 17.6 do chapra)

# ajusta uma reta y = a0 + a1 x num conjunto de pontos que tem erro/dispersão (diferente da interpolação,
# aqui a reta n precisa passar por cima dos pontos, ela só tem que ficar o mais perto possível de todos)
# o critério é minimizar a soma dos quadrados dos resíduos: Sr = soma de (y_i - a0 - a1 x_i)²
# derivando Sr em relação a a0 e a a1 e igualando a zero sai um sistema 2x2 (equações normais), que já tem solução fechada:
# a1 = (n soma(xy) - soma(x) soma(y)) / (n soma(x²) - soma(x)²)
# a0 = ym - a1 xm

# x, y: pontos (x1, y1) ... (xn, yn)
# n: número de pontos
# a1: inclinação da reta
# a0: intercepto (valor de y quando x = 0)
# syx: erro padrão da estimativa. É o "desvio padrão" dos pontos em volta da reta, syx = raiz(Sr / (n - 2))
# r2: coeficiente de determinação. Fração da variação dos y que a reta explica. r2 = (St - Sr) / St
# info: 0 = ok, 1 = todos os x iguais (denominador do a1 é 0, n existe reta única)

# St = soma dos quadrados em volta da média (erro de usar só ym pra prever y). Sr = soma dos quadrados em volta da reta
# r2 = 1 -> a reta passa em cima de todos os pontos. r2 próximo de 0 -> a reta n melhora nada em relação à média
# o coeficiente de correlação é r = raiz(r2), mas o sinal dele depende do sinal de a1 (esse algoritmo n devolve o r)

# ps: o pseudocódigo do livro é 1-based, python é 0-based, então os loops vão de 0 até n-1
# ps2: com n = 2 a reta passa exata nos 2 pontos e o syx n é definido (divide por n-2 = 0), então ele volta nan

import numpy as np

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
    if den == 0:
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

# main
# exemplo 17.1 do chapra: ajuste de uma reta nos 7 pontos abaixo
# resposta esperada: y = 0,07142857 + 0,8392857 x, syx ≈ 0,7735 e r2 ≈ 0,868
x = [1, 2, 3, 4, 5, 6, 7]
y = [0.5, 2.5, 2.0, 4.0, 3.5, 6.0, 5.5]

a1, a0, syx, r2, info = regress(x, y)

print("a1 =", a1)
print("a0 =", a0)
print("y = {:.7f} + {:.7f} x".format(a0, a1))
print("syx =", syx)
print("r2 =", r2)
print("r =", np.sign(a1) * np.sqrt(r2)) # coeficiente de correlação, com o sinal de a1
print("info =", info)

# confere com a lib
print("\n(a1, a0) (np.polyfit) =", np.polyfit(x, y, 1))
print("r2 (np.corrcoef) =", np.corrcoef(x, y)[0, 1] ** 2)

# todos os x iguais: n existe reta. Tem que avisar e voltar info = 1
print("\nteste x todos iguais:")
_, _, _, _, info2 = regress([2, 2, 2], [1, 2, 3])
print("info =", info2)