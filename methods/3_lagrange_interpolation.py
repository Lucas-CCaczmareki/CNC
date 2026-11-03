## implementação do polinômio interpolador de Lagrange (fig 18.11 do chapra)

# mesmo objetivo do newton: dados n+1 pontos (x_i, y_i), estimar o valor da função num x que n tá na tabela
# a diferença é a forma de montar o polinômio. Aqui n tem tabela de diferenças divididas, cada ponto vira uma "parcela":
# p_n(x) = soma de y_i * L_i(x), onde L_i(x) = produto de (x - x_j)/(x_i - x_j), com j != i
# o L_i(x) é o polinômio de Lagrange (ou função base). Ele vale 1 em x = x_i e 0 em todos os outros x_j
# então cada parcela "liga" só o seu ponto, e a soma passa por todos eles

# o polinômio que sai é o MESMO do newton (o polinômio de grau n que passa por n+1 pontos é único), só muda a conta
# desvantagem: n dá pra reaproveitar nada. Se chegar um ponto novo, refaz tudo. E também n tem estimativa de erro de graça (o newton tem o ea)
# vantagem: o algoritmo é bem simples e n precisa guardar tabela nenhuma. Bom pra calcular uma estimativa só

# x, y: vetores com os pontos conhecidos (x n precisa estar ordenado, mas n pode ter x repetido senão divide por 0)
# n: grau do polinômio (n + 1 pontos)
# xi: ponto onde a gente quer interpolar
# retorna o valor do polinômio em xi

# ps: o pseudocódigo do livro chama o ponto de interpolação de "x", aqui chamei de xi pra n bater com o vetor x dos dados

import numpy as np

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
# mesmo exemplo do newton (chapra ex 18.4): estimar ln(2) com os pontos de ln(x) em x = 1, 4, 6, 5. Valor verdadeiro: 0,6931472
# aqui n tem o ea de cada ordem, então a gente roda 1 vez pra cada grau pra comparar com o newton
# resposta esperada (livro): grau 1 = 0,4620981 | grau 2 = 0,5658444 | grau 3 = 0,6287686
# aqui dá 0,462098 | 0,5658442 | 0,6287674 (diferença na 6a casa pq o y tá arredondado em 6 dígitos, igual no newton)
x = [1, 4, 6, 5]
y = [0, 1.386294, 1.791759, 1.609438]
xi = 2

print("grau  lagrange  erro verdadeiro")
verdadeiro = np.log(xi)
for n in range(1, 4):
    # pro grau n usa só os n + 1 primeiros pontos da lista (mesma ideia do newton, que adiciona 1 ponto por ordem)
    p = lagrange(x[:n + 1], y[:n + 1], n, xi)
    print(n, round(float(p), 7), round(float(verdadeiro - p), 7))

# confere com a lib (np.polyfit com grau n passando por todos os pontos dá o mesmo polinômio, pq ele é único)
coef = np.polyfit(x, y, 3)
print("\nlagrange (grau 3) =", lagrange(x, y, 3, xi))
print("np.polyfit (grau 3) =", np.polyval(coef, xi))

# o resultado tem que ser igual ao do interpolacao_newton.py, e o polinômio passa exatamente nos pontos dados:
print("passa nos pontos? max|p(x_i) - y_i| =", max(abs(lagrange(x, y, 3, xk) - yk) for xk, yk in zip(x, y)))